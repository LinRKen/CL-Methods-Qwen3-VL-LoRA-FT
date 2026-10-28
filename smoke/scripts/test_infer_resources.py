'''CPU resource and SIGTERM acceptance. No model imports or real GPU locks.'''
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import test_infer_executor as fixture
ex=fixture.ex

def eventually(predicate,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.05)
    raise AssertionError('Condition did not become true before deadline')

def child_executor(job_dir,case):
    original=fixture.REAL_EXCLUSIVE
    @contextmanager
    def locks(path,blocking=False):
        path=Path(path)
        if str(path).startswith('/tmp/qwen3vl-lora-tune-locks'):path=case/'fake-locks'/path.name
        with original(path,blocking=blocking):yield
    def popen(command,**kwargs):
        kwargs['env']['CUDA_VISIBLE_DEVICES']=''
        return fixture.REAL_POPEN([sys.executable,'-B',__file__,'--sleeper',str(case/'worker-pids.json')],**kwargs)
    try:
        with patch.object(ex,'_exclusive',side_effect=locks),patch.object(ex,'_gpu_identity',return_value={'physical_ids':[1],'uuids':['CPU-FAKE-1'],'pci_bus_ids':['00000000:03:00.0']}),patch.object(ex.subprocess,'Popen',side_effect=popen):
            ex.execute_infer_job(job_dir)
    except InterruptedError:return 17
    return 0

if len(sys.argv)>1 and sys.argv[1]=='--executor-child':raise SystemExit(child_executor(Path(sys.argv[2]),Path(sys.argv[3])))
if len(sys.argv)>1 and sys.argv[1]=='--sleeper':
    worker=fixture.REAL_POPEN([sys.executable,'-B','-c','import time; time.sleep(120)'])
    def terminate(number,frame):
        worker.terminate(); worker.wait(timeout=5); raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    fixture.write_json_atomic(Path(sys.argv[2]),{'worker':os.getpid(),'descendant':worker.pid,'pgid':os.getpgrp()})
    time.sleep(120)
    worker.terminate(); worker.wait(); raise SystemExit(0)
if len(sys.argv)>1 and sys.argv[1]=='--slot-child':
    with ex._stage_slot(Path(sys.argv[2]),1):Path(sys.argv[3]).write_text('acquired')
    raise SystemExit(0)

class ResourceTests(fixture.InferTests):
    def launch(self,job):
        log=(self.case/'controller.log').open('xb'); self.addCleanup(log.close)
        process=fixture.REAL_POPEN([sys.executable,'-B',__file__,'--executor-child',str(job),str(self.case)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def clean():
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        self.addCleanup(clean); return process
    def read_receipt(self,job):
        path=job/'execution.json'
        return fixture.read_json(path) if path.exists() else {}
    def test_job_lock_rejects_duplicate_without_changes(self):
        _,_,jobs=self.records(); job=jobs['infer'][0]
        with fixture.REAL_EXCLUSIVE(job/'.execution.lock'):
            with self.assertRaisesRegex(ValueError,'Another executor'):self.execute(job)
        self.assertFalse((job/'execution.json').exists()); self.assertEqual(fixture.load_job(job)[0]['status'],'planned')
    def test_stage_slot_capacity_blocks_then_releases(self):
        _,runs,_=self.records(); ready=self.case/'slot-ready'
        with ex._stage_slot(runs['infer'],1) as index:
            self.assertEqual(index,0)
            process=fixture.REAL_POPEN([sys.executable,'-B',__file__,'--slot-child',str(runs['infer']),str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                time.sleep(.4); self.assertIsNone(process.poll()); self.assertFalse(ready.exists())
            except BaseException:process.terminate();process.wait();raise
        _,error=process.communicate(timeout=15); self.assertEqual(process.returncode,0,error); self.assertEqual(ready.read_text(),'acquired')
    def test_cancel_while_waiting_shared_gpu_lock(self):
        _,runs,jobs=self.records(); job=jobs['infer'][0]; lock=self.case/'fake-locks/physical-1.lock'
        with fixture.REAL_EXCLUSIVE(lock):
            process=self.launch(job); eventually(lambda:self.read_receipt(job).get('status')=='preparing'); time.sleep(.3)
            self.assertFalse((self.case/'worker-pids.json').exists()); self.assertEqual(fixture.load_job(job)[0]['status'],'planned')
            process.send_signal(signal.SIGTERM); self.assertEqual(process.wait(timeout=20),17)
            self.assertFalse(Path(fixture.load_job(job)[0]['output_dir']).exists())
        receipt=self.read_receipt(job); self.assertEqual(receipt['status'],'cancelled'); self.assertTrue(receipt['cleanup_completed'])
        self.assertEqual(fixture.load_job(job)[0]['status'],'cancelled')
        with fixture.REAL_EXCLUSIVE(lock):pass
        with ex._stage_slot(runs['infer'],1):pass
    def test_sigterm_cleans_worker_descendant_and_locks(self):
        _,runs,jobs=self.records(); job=jobs['infer'][0]; process=self.launch(job); evidence=self.case/'worker-pids.json'; eventually(evidence.exists)
        pids=fixture.read_json(evidence); self.assertEqual(pids['pgid'],pids['worker']); self.assertEqual(self.read_receipt(job)['launcher_pid'],pids['worker'])
        process.send_signal(signal.SIGTERM); self.assertEqual(process.wait(timeout=20),17)
        eventually(lambda:all(not Path('/proc/'+str(pids[k])).exists() for k in ('worker','descendant')))
        record,_=fixture.load_job(job); receipt=self.read_receipt(job)
        self.assertEqual(record['status'],'cancelled'); self.assertEqual(record['artifacts'],[]); self.assertEqual(receipt['status'],'cancelled'); self.assertTrue(receipt['cleanup_completed'])
        self.assertTrue(Path(receipt['output_dir']).exists()); self.assertFalse((Path(receipt['output_dir'])/'inference_result.json').exists())
        with fixture.REAL_EXCLUSIVE(job/'.execution.lock'):pass
        with fixture.REAL_EXCLUSIVE(self.case/'fake-locks/physical-1.lock'):pass
        with ex._stage_slot(runs['infer'],1):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result-root',required=True,type=Path);args=parser.parse_args();fixture.RESULT=args.result_root.resolve();fixture.RESULT.mkdir(parents=True,exist_ok=False)
    suite=unittest.TestSuite(ResourceTests(n) for n in sorted(vars(ResourceTests)) if n.startswith('test_'));result=unittest.TextTestRunner(verbosity=2).run(suite)
    fixture.write_json_atomic(fixture.RESULT/'summary.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'details':[{'test':str(t),'traceback':s} for t,s in result.failures+result.errors]});return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
