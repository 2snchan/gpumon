#!/usr/bin/env python3

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ===== Fixed constants =====
CONCURRENCY = 2
JITTER = 0.4
TIMEOUT = 10.0
CONNECT_TIMEOUT = 4.0
NVIDIA_SMI = "nvidia-smi"
SSH_OPT = ""

CSV_OPTS = "--format=csv,noheader,nounits"

# GPU query (uuid for per-GPU user mapping)
QUERY_GPU = (
    "--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total,driver_version"
)

# CUDA version (some drivers don't expose via query)
CUDA_PARSE_CMD = r"nvidia-smi -q | sed -n 's/.*CUDA Version *: *\([0-9.]\+\).*/\1/p' | head -n1"

# Compute apps: per-GPU top user resolution
APPS_QUERY = "--query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"

# Config file path (~/.gpumon)
CONFIG_PATH = Path.home() / ".gpumon"


# ============================= Config I/O =============================

def load_config() -> Dict[str, object]:
    """Load JSON config from ~/.gpumon. Keys we use:
       - hosts_content: str (the actual content of hosts file)
       - interval: float
       - ssh_key: str
       - blacklist: str
       - initial_wait: int
       - last_used_hosts_path: str (optional, for convenience)
    """
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def save_config(cfg: Dict[str, object]) -> None:
    """Persist JSON config to ~/.gpumon (0600 on POSIX)."""
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if os.name == "posix":
            with contextlib.suppress(Exception):
                os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        # Not fatal
        pass


# ============================= Helpers =============================

def parse_hosts_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def parse_hosts_text(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

def which_ssh() -> str:
    return "ssh.exe" if platform.system().lower().startswith("win") else "ssh"

def build_ssh_cmd(
    host: str,
    remote_cmd: str,
    identity: str = "",
    connect_timeout: float = CONNECT_TIMEOUT,
) -> List[str]:
    ssh_bin = which_ssh()
    extra_opts = shlex.split(SSH_OPT) if SSH_OPT else []
    base_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={int(max(1, connect_timeout))}",
        "-o", "PreferredAuthentications=publickey",
        "-o", "PubkeyAuthentication=yes",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=1",
        "-o", "ConnectionAttempts=1",
        "-o", "TCPKeepAlive=yes",
        "-o", "LogLevel=ERROR",
    ]
    id_part: List[str] = []
    if identity:
        id_part = ["-i", identity, "-o", "IdentitiesOnly=yes"]
    return [ssh_bin] + base_opts + id_part + extra_opts + [host, remote_cmd]

def short_host(label: str) -> str:
    s = label
    if "@" in s:
        s = s.split("@", 1)[1]
    if "." in s:
        s = s.split(".", 1)[0]
    return s

def parse_blacklist(spec: str) -> Dict[str, Set[int]]:
    bl: Dict[str, Set[int]] = {}
    if spec is None or spec == "":
        return bl
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" in group:
            host_key, idxs = group.split(":", 1)
        else:
            host_key, idxs = "*", group
        idx_set: Set[int] = set()
        for tok in idxs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx_set.add(int(tok))
            except ValueError:
                pass
        if idx_set:
            bl.setdefault(host_key, set()).update(idx_set)
    return bl

def gpu_idx_blacklisted(host_label: str, idx: int, bl: Dict[str, Set[int]]) -> bool:
    if idx in bl.get("*", set()):
        return True
    if idx in bl.get(host_label, set()):
        return True
    for key, s in bl.items():
        if key in ("*", host_label) or not key:
            continue
        if key in host_label and idx in s:
            return True
    return False


# ============================= Styling =============================

def colorize_temp(t: Optional[int]) -> Text:
    if t is None:
        return Text("-", style="dim")
    if t >= 75:
        return Text(f"{t:>3}", style="bold red")
    elif t >= 50:
        return Text(f"{t:>3}", style="yellow")
    elif t >= 40:
        return Text(f"{t:>3}", style="white")
    else:
        return Text(f"{t:>3}", style="dim")

def colorize_host(h: str, stale_age: Optional[float], has_error: bool) -> Text:
    h_fixed = f"{h:<9}"
    if has_error and stale_age is None:
        return Text(h_fixed, style="bold red")
    if stale_age is None:
        return Text(h_fixed, style="bold")
    if stale_age < 10:
        return Text(h_fixed, style="bold")
    elif stale_age < 30:
        return Text(h_fixed, style="bold yellow")
    else:
        return Text(h_fixed, style="bold red")

def percent_style(kind: str, p: float) -> str:
    if kind == "util":
        if p >= 80: return "red"
        if p >= 20: return "yellow"
        if p >= 10: return "white"
        return "dim"
    else:
        if p >= 80: return "red"
        if p >= 20: return "yellow"
        if p >= 10: return "white"
        return "dim"

def fixed_pct(p: float) -> str:
    return f"{int(round(p)):>3}%"

def fixed_mib_pair(used: int, total: int) -> str:
    return f"{used:>6}/{total:<6} MiB"


# ============================= Bars (original logic kept) =============================

def decide_bar_len(console_width: int, name_avg_len: int = 22) -> int:
    fixed_est = 43
    text_payload_est = 25
    free = max(0, console_width - fixed_est - name_avg_len - text_payload_est)
    if free < 6:
        return 0
    if free < 18:
        return 6
    if free < 30:
        return 10
    if free < 42:
        return 14
    if free < 60:
        return 18
    return 22

def draw_bar(pct: float, length: int, style: str) -> Text:
    length = max(3, length)
    fill = int(round((pct / 100.0) * length))
    fill = max(0, min(length, fill))
    return Text("█" * fill + "░" * (length - fill), style=style)

def format_util_mem(util: Optional[float], mem_used: int, mem_total: int, console_width: int) -> Tuple[Text, Text]:
    bar_len = decide_bar_len(console_width)
    util_pct = 0.0 if (util is None or math.isnan(util)) else max(0.0, min(100.0, float(util)))
    util_style = percent_style("util", util_pct)
    util_pct_text = Text(fixed_pct(util_pct), style=util_style)

    mem_pct = 0.0 if mem_total <= 0 else max(0.0, min(100.0, (mem_used / mem_total) * 100.0))
    mem_style = percent_style("mem", mem_pct)
    mem_pct_text = Text(fixed_pct(mem_pct), style=mem_style)
    mem_abs_text = Text(f" {fixed_mib_pair(mem_used, mem_total)}", style="white")

    if bar_len > 0:
        util_bar = draw_bar(util_pct, bar_len, util_style)
        mem_bar = draw_bar(mem_pct, bar_len, mem_style)
        util_cell = Text.assemble(util_bar, Text(" "), util_pct_text)
        mem_cell = Text.assemble(mem_bar, Text(" "), mem_pct_text, mem_abs_text)
    else:
        util_cell = util_pct_text
        mem_cell = Text.assemble(mem_pct_text, Text(" "), mem_abs_text)
    return util_cell, mem_cell


# ============================= SSH / Queries =============================

async def ssh_run(host: str, cmd: str, identity: str) -> Tuple[int, str, str, float]:
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *build_ssh_cmd(host, cmd, identity, connect_timeout=CONNECT_TIMEOUT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        dt = time.time() - t0
        return proc.returncode, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore"), dt
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 255, "", f"timeout({TIMEOUT:.0f}s)", TIMEOUT

async def fetch_gpu_and_cuda(host: str, wait_first: bool, wait_sec: int, identity: str) -> Tuple[int, str, str, Optional[str]]:
    gpu_cmd = f"LC_ALL=C {shlex.quote(NVIDIA_SMI)} {QUERY_GPU} {CSV_OPTS}"
    if wait_first and wait_sec > 0:
        gpu_cmd = f"sleep {wait_sec}; {gpu_cmd}"
    rc, out, err, _ = await ssh_run(host, gpu_cmd, identity)

    cuda_rc, cuda_out, cuda_err, _ = await ssh_run(host, CUDA_PARSE_CMD, identity)
    cuda_ver = None
    if cuda_rc == 0:
        v = cuda_out.strip().splitlines()
        if v:
            cuda_ver = v[0].strip() or None
    return rc, out, err, cuda_ver

async def fetch_apps_top_users_per_gpu(host: str, identity: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    rc, out, err, _ = await ssh_run(host, f"LC_ALL=C {NVIDIA_SMI} {APPS_QUERY}", identity)
    top_pid_by_uuid: Dict[str, int] = {}
    if rc != 0 or not out.strip():
        return top_pid_by_uuid, {}
    top_mem_by_uuid: Dict[str, int] = {}
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        try:
            pid = int(pid_s); mem = int(mem_s)
        except Exception:
            continue
        if uuid not in top_mem_by_uuid or mem > top_mem_by_uuid[uuid]:
            top_mem_by_uuid[uuid] = mem
            top_pid_by_uuid[uuid] = pid

    if not top_pid_by_uuid:
        return top_pid_by_uuid, {}
    pid_list = ",".join(str(p) for p in top_pid_by_uuid.values())
    rc2, out2, err2, _ = await ssh_run(host, f"ps -o pid=,user= -p {pid_list}", identity)
    user_by_pid: Dict[int, str] = {}
    if rc2 == 0 and out2.strip():
        for ln in out2.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                pid_str, user = ln.split(None, 1)
                user_by_pid[int(pid_str)] = user.strip()
            except Exception:
                continue
    return top_pid_by_uuid, user_by_pid


# ============================= Parsing =============================

def parse_gpu_rows(text: str):
    rows = []
    driver = None
    uuid_by_index: Dict[int, str] = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 8:
            continue
        try:
            idx = int(parts[0]); uuid = parts[1]; name = parts[2]
            util = int(parts[3]); temp = int(parts[4])
            mem_used = int(parts[5]); mem_total = int(parts[6])
            driver = parts[7]
            uuid_by_index[idx] = uuid
            rows.append({
                "idx": idx, "uuid": uuid, "name": name, "util": util, "temp": temp,
                "mem_used": mem_used, "mem_total": mem_total,
            })
        except Exception:
            pass
    return rows, driver, uuid_by_index


# ============================= Rendering (original layout) =============================

def render_table(state: Dict[str, dict], interval: float, now_ts: float, console_width: int) -> Table:
    tbl = Table(title=f"Multi-Server GPU Monitor", box=box.SIMPLE_HEAVY)
    tbl.add_column("Host", no_wrap=True)
    tbl.add_column("CUDA", no_wrap=True)
    tbl.add_column("GPU", justify="right", no_wrap=True)
    tbl.add_column("User", no_wrap=True)
    tbl.add_column("Name", no_wrap=False)
    tbl.add_column("Util", justify="left", no_wrap=False)
    tbl.add_column("Mem", justify="left", no_wrap=False)
    tbl.add_column("Temp", justify="right", no_wrap=True)

    for host_key, data in state.items():
        sh = short_host(host_key)
        last_ok_ts = data.get("last_ok_ts")
        has_error = bool(data.get("error"))
        stale_age = None if last_ok_ts is None else (now_ts - last_ok_ts)
        host_cell = colorize_host(sh, stale_age, has_error)
        cuda_text = Text(f"{(data.get('cuda') or '-'):>5}", style="slate_blue3")

        gpus = data.get("gpus", [])
        top_user_by_idx: Dict[int, str] = data.get("top_user_by_idx", {}) or {}
        if not gpus:
            tbl.add_row(host_cell, cuda_text, "-", "-", "-", "-", "-", "-")
            continue

        first = True
        for r in sorted(gpus, key=lambda x: x["idx"]):
            util_cell, mem_cell = format_util_mem(float(r["util"]), r["mem_used"], r["mem_total"], console_width)
            temp_cell = colorize_temp(r["temp"])
            name_cell = Text(r["name"], style="cyan")
            user_cell = Text(top_user_by_idx.get(r["idx"]) or "-", style="green3")

            tbl.add_row(
                host_cell if first else Text(" " * 9),
                cuda_text if first else Text(" " * 5),
                f"{r['idx']:>2}",
                user_cell,
                name_cell,
                util_cell,
                mem_cell,
                temp_cell,
            )
            first = False
    return tbl


# ============================= Change detection =============================

def _band_key(last_ok_ts: Optional[float], has_error: bool, now_ts: float) -> str:
    if has_error and last_ok_ts is None:
        return "ERR"
    if last_ok_ts is None:
        return "OK"
    age = now_ts - last_ok_ts
    if age < 10:
        return "OK"
    if age < 30:
        return "STALE1"
    return "STALE2"

def build_digest(state: Dict[str, dict], now_ts: float) -> Tuple:
    out = []
    for host_key in sorted(state.keys()):
        d = state[host_key]
        band = _band_key(d.get("last_ok_ts"), bool(d.get("error")), now_ts)
        cuda = d.get("cuda")
        gpus = tuple(
            (r["idx"], r["name"], r["util"], r["mem_used"], r["mem_total"], r["temp"])
            for r in sorted(d.get("gpus", []), key=lambda x: x["idx"])
        )
        users_t = tuple(sorted((d.get("top_user_by_idx") or {}).items()))
        out.append((short_host(host_key), band, cuda, gpus, users_t))
    return tuple(out)


# ============================= Main =============================

async def _amain(args: argparse.Namespace) -> None:
    # clear once and print a blank line
    console.clear()
    console.print()

    # we get hosts_lines directly (already parsed)
    hosts = args.hosts_lines
    if not hosts:
        console.print("[red]No hosts found[/red]")
        sys.exit(1)

    blspec = parse_blacklist(args.blacklist)
    state: Dict[str, dict] = {
        h: {"gpus": [], "last_ok_ts": None, "error": None, "warmed": False,
            "cuda": None, "driver": None, "top_user_by_idx": {}} for h in hosts
    }
    prev_digest: Optional[Tuple] = None

    sem = asyncio.Semaphore(max(1, CONCURRENCY))

    async def poll_host(h: str):
        async with sem:
            if not state[h]["warmed"] and JITTER > 0:
                await asyncio.sleep(random.random() * JITTER)

            rc, out, err, cuda_ver = await fetch_gpu_and_cuda(
                host=h, wait_first=not state[h]["warmed"], wait_sec=args.initial_wait, identity=args.ssh_key
            )
            if rc == 0 and out.strip():
                rows, driver, uuid_by_index = parse_gpu_rows(out)
                base = short_host(h)
                rows = [r for r in rows if not gpu_idx_blacklisted(base, r["idx"], blspec)]

                top_pid_by_uuid, user_by_pid = await fetch_apps_top_users_per_gpu(h, args.ssh_key)
                top_user_by_idx: Dict[int, str] = {}
                for idx, uuid in uuid_by_index.items():
                    if uuid in top_pid_by_uuid:
                        pid = top_pid_by_uuid[uuid]
                        user = user_by_pid.get(pid)
                        if user:
                            top_user_by_idx[idx] = user

                state[h]["gpus"] = rows
                state[h]["last_ok_ts"] = time.time()
                state[h]["error"] = None
                state[h]["warmed"] = True
                if driver:
                    state[h]["driver"] = driver
                if cuda_ver:
                    state[h]["cuda"] = cuda_ver
                if top_user_by_idx:
                    state[h]["top_user_by_idx"] = top_user_by_idx
            else:
                last_line = (err.strip().splitlines() or [""])[-1] if err else f"rc={rc}"
                state[h]["error"] = last_line

    with Live(
        Panel("Initializing...", border_style="cyan"),
        refresh_per_second=24,
        console=console,
        auto_refresh=False,  # only update when digest changes
    ) as live:
        while True:
            tasks = [poll_host(h) for h in hosts]
            await asyncio.gather(*tasks, return_exceptions=True)
            now_ts = time.time()
            width = console.size.width

            digest = build_digest(state, now_ts)
            if digest != prev_digest:
                prev_digest = digest
                live.update(render_table(state, args.interval, now_ts, width), refresh=True)

            await asyncio.sleep(args.interval)

def run() -> None:
    ap = argparse.ArgumentParser(description="Multi-host GPU terminal monitor")
    # All optional — if omitted, read ~/.gpumon
    ap.add_argument("-H", "--hosts", default=None, help="hosts file path (alias/fqdn/user@host each line)")
    ap.add_argument("-i", "--interval", type=float, default=None, help="refresh interval seconds")
    ap.add_argument("--ssh-key", default=None, help="identity file path (e.g. ~/.ssh/id_ed25519)")
    ap.add_argument("--blacklist", default=None, help="GPU blacklist, e.g. 'monday:0,1;gloryday:2;*:7'")
    ap.add_argument("--initial-wait", type=int, default=None, help="per-host first-connection wait seconds")
    args_in = ap.parse_args()

    # 1) Load existing config
    cfg = load_config()

    # 2) Resolve hosts content (prefer CLI)
    hosts_content: Optional[str] = None
    last_used_path = None
    if args_in.hosts:
        last_used_path = args_in.hosts
        try:
            hosts_content = Path(args_in.hosts).read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Failed to read hosts file:[/red] {e}")
            sys.exit(1)
    else:
        hosts_content = cfg.get("hosts_content")

    if not hosts_content:
        console.print("[red]No hosts content provided and no ~/.gpumon config found.[/red]")
        console.print("Usage: gpumon -H hosts.txt [--ssh-key KEY] [--blacklist SPEC] [--initial-wait N] [-i SECS]")
        sys.exit(1)

    hosts_lines = parse_hosts_text(hosts_content)
    if not hosts_lines:
        console.print("[red]Parsed hosts list is empty.[/red]")
        sys.exit(1)

    # 3) Merge other args with config (CLI precedes)
    interval  = args_in.interval if args_in.interval is not None else cfg.get("interval", 1)
    ssh_key   = args_in.ssh_key if args_in.ssh_key is not None else cfg.get("ssh_key", "")
    blacklist = args_in.blacklist if args_in.blacklist is not None else cfg.get("blacklist", "")
    init_wait = args_in.initial_wait if args_in.initial_wait is not None else cfg.get("initial_wait", 5)

    # 4) Save merged config (store hosts_content, not just path)
    save_config({
        "hosts_content": hosts_content,
        "interval": interval,
        "ssh_key": ssh_key,
        "blacklist": blacklist,
        "initial_wait": init_wait,
        "last_used_hosts_path": last_used_path or cfg.get("last_used_hosts_path", ""),
    })

    # 5) Run with resolved arguments
    merged = argparse.Namespace(
        hosts_lines=hosts_lines,
        interval=interval,
        ssh_key=ssh_key,
        blacklist=blacklist,
        initial_wait=init_wait,
    )
    try:
        asyncio.run(_amain(merged))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()

