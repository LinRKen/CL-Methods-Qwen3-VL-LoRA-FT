'''Resource and interruption tests for eval; only fake workers and isolated locks.'''
import argparse,json,os,signal,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
import test_eval_executor as fixture
ex=fixture.ex
SELF=Path(__file__).resolve()

def eventually(f,timeout=20):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if f():return
        time.sleep(.05)
    raise AssertionError('Timeout waiting for smoke evidence')

def controller(job,case):
    def lock(path,**kw):
        path=Path(path)
        if path.parent==Path('/tmp/qwen3vl-lora-tune-locks'):path=case/'fake-locks'/path.name
        return fixture.REAL_LOCK(path,**kw)
    def spawn(command,**kw):
        return fixture.REAL_POPEN([sys.executable,'-B',str(SELF),'--sleeper',str(case/'worker-pids.json')],**{**kw,'env':{**kw['env'],'CUDA_VISIBLE_DEVICES':''}})
    try:
        with patch.object(ex,'_exclusive',side_effect=lock),patch.object(ex,'_gpu_identity',side_effect=fixture.hi.identity),patch.object(ex.subprocess,'Popen',side_effect=spawn):
            ex.execute_eval_job(job,input_manifests=fixture.read_json(case/'resource-inputs.json'))
    except (InterruptedError,KeyboardInterrupt):return 130
    return 0

class ResourceTests(fixture.EvalTests):
    def launch(self,job,inputs):
        fixture.write_json_atomic(self.case/'resource-inputs.json',inputs)
        log=(self.case/'controller.log').open('xb');self.addCleanup(log.close)
        p=fixture.REAL_POPEN([sys.executable,'-B',str(SELF),'--controller',str(job),str(self.case)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def cleanup():
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        self.addCleanup(cleanup);return p
    def receipt(self,d):return fixture.read_json(d/'execution.json') if (d/'execution.json').exists() else {}
    def test_stage_capacity_blocks_then_releases(self):
        _,r,_=self.prepare();run=r['eval'];ready=self.case/'slot-acquired'
        with ex._stage_slot(run,1) as n:
            self.assertEqual(n,0)
            p=fixture.REAL_POPEN([sys.executable,'-B',str(SELF),'--slot',str(run),str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            self.addCleanup(lambda:p.poll() is None and p.terminate())
            time.sleep(.5);self.assertIsNone(p.poll());self.assertFalse(ready.exists())
        _,error=p.communicate(timeout=15);self.assertEqual(p.returncode,0,error);self.assertEqual(ready.read_text(),'acquired')
    def test_cancel_waiting_gpu_lock(self):
        _,r,_=self.prepare();d=r['eval']/'jobs/eval-judge-all';lock=self.case/'fake-locks/physical-1.lock'
        with fixture.REAL_LOCK(lock):
            p=self.launch(d,{})
            eventually(lambda:self.receipt(d).get('status')=='preparing');time.sleep(.4);p.send_signal(signal.SIGTERM);self.assertEqual(p.wait(timeout=20),130)
            self.assertFalse(Path(fixture.load_job(d)[0]['output_dir']).exists())
        receipt=self.receipt(d);self.assertEqual(receipt['status'],'cancelled');self.assertTrue(receipt['cleanup_completed']);self.assertEqual(fixture.load_job(d)[0]['status'],'cancelled')
        with fixture.REAL_LOCK(lock):pass
        with ex._stage_slot(r['eval'],1):pass
    def test_sigterm_cleans_worker_descendant_and_locks(self):
        d,inputs,r,_=self.ready();p=self.launch(d,inputs);evidence=self.case/'worker-pids.json';eventually(evidence.exists);pids=fixture.read_json(evidence)
        self.assertEqual(pids['pgid'],pids['worker']);self.assertEqual(self.receipt(d)['launcher_pid'],pids['worker'])
        p.send_signal(signal.SIGTERM);self.assertEqual(p.wait(timeout=25),130)
        eventually(lambda:all(not Path('/proc/'+str(pids[k])).exists() for k in ('worker','descendant')))
        job,_=fixture.load_job(d);receipt=self.receipt(d);self.assertEqual(job['status'],'cancelled');self.assertEqual(job['artifacts'],[]);self.assertTrue(receipt['cleanup_completed']);self.assertFalse((Path(job['output_dir'])/'evaluation_result.json').exists())
        with fixture.REAL_LOCK(d/'.execution.lock'):pass
        with fixture.REAL_LOCK(self.case/'fake-locks/physical-1.lock'):pass
        with ex._stage_slot(r['eval'],1):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--result-root',required=True,type=Path);a=p.parse_args();root=a.result_root.resolve();root.mkdir(parents=True,exist_ok=False)
    if not root.is_relative_to((fixture.hi.REPO/'smoke/results').resolve()):p.error('smoke results only')
    fixture.RESULT=fixture.hi.RESULT=root
    suite=unittest.TestSuite(ResourceTests(n) for n in sorted(ResourceTests.__dict__) if n.startswith('test_'));r=unittest.TextTestRunner(verbosity=2).run(suite)
    fixture.write_json_atomic(root/'summary.json',{'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'successful':r.wasSuccessful(),'details':[{'test':str(t),'traceback':s} for t,s in r.failures+r.errors]});return 0 if r.wasSuccessful() else 1
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--controller':raise SystemExit(controller(Path(sys.argv[2]),Path(sys.argv[3])))
    if len(sys.argv)>1 and sys.argv[1]=='--slot':
        with ex._stage_slot(Path(sys.argv[2]),1):Path(sys.argv[3]).write_text('acquired')
        raise SystemExit(0)
    if len(sys.argv)>1 and sys.argv[1]=='--sleeper':
        child=fixture.REAL_POPEN([sys.executable,'-B','-c','import time; time.sleep(120)'],env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def stop(n,f):
            child.terminate();child.wait(timeout=5);raise SystemExit(0)
        signal.signal(signal.SIGTERM,stop);fixture.write_json_atomic(Path(sys.argv[2]),{'worker':os.getpid(),'descendant':child.pid,'pgid':os.getpgrp()});time.sleep(120);stop(None,None)
    raise SystemExit(main())

