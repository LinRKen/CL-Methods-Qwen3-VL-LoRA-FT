"""Opt-in early interpreter audit, used only by the smoke launcher."""
import json
import os
path=os.environ.get('EVAL_SMOKE_ENV_AUDIT')
if path:
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    permitted=('1','GPU-63a25b73-5c01-919f-188a-29c116904574')
    row={'pid':os.getpid(),'ppid':os.getppid(),'CUDA_VISIBLE_DEVICES':visible,
         'CUDA_DEVICE_ORDER':os.environ.get('CUDA_DEVICE_ORDER'),
         'VLLM_WORKER_MULTIPROC_METHOD':os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
    try: os.write(fd,(json.dumps(row)+'\n').encode())
    finally: os.close(fd)
    if visible not in permitted:
        os.write(2,b'Smoke audit rejected a worker GPU mask\n')
        os._exit(72)
