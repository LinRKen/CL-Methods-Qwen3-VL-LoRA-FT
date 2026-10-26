#!/usr/bin/env python3
"""Inject a controlled interruption after a completed batch, leaving valid raw rows.

Accepts the unchanged production eval_cosine.py arguments. All semantic scoring,
file writes, engine loading and shutdown remain production functions.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/COSINE'))
import eval_cosine as cli

class SmokeInterruption(Exception):
    pass

if __name__=='__main__':
    target=int(os.environ.get('SMOKE_INTERRUPT_AFTER','64'))
    original=cli.judge_batch
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        done=kwargs.get('done',args[7] if len(args)>7 else {})
        if len(done)>=target:
            raise SmokeInterruption('Controlled smoke interruption after {} fully written rows'.format(len(done)))
    cli.judge_batch=interrupted
    try:
        cli.main()
    except SmokeInterruption as error:
        print(str(error),flush=True)
        sys.exit(75)
