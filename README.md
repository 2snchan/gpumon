# gpumon-ssh

Multi-host GPU monitor over SSH. Developed by Seungchan Lee

## Install (local)
```bash
pip install .
```

## Usage
host.txt file needed for first usage. After first execution, .gpumon file automatically generated in your home folder.
```
mike.lee@badday.yonsei.ac.kr
mike.lee@gpuserver.com
```
```bash
gpumon -H Host.txt
```
If you want to add some gpus in blacklist, you can pass --blacklist "servername:CUDANUMBER,NUMBER,NUMBER", while servername is top-level domain of server address.
