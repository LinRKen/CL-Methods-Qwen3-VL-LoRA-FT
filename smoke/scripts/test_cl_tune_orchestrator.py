"""CPU CL acceptance: real frozen records/executors, simulated training workers."""
import argparse,json,os,subprocess,sys,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import test_tune_orchestrator as b
from lib import tune_executor as ex,tune_orchestrator as orch
from lib.common import TASKS,read_json,write_json_atomic
from lib.records import load_job,record_status
from lib.config import load_config

class CLTests(unittest.TestCase):
    setUp=b.T.setUp
    config=b.T.config
    execute=b.T.execute
    events=b.T.events
    peak=b.T.peak
    reject=b.T.reject
    def prepare(self,extra=None): return b.T.prepare(self,'cl',extra)
    def prefix(self,run,n):
        previous=None; jobs=[]
        for i,t in enumerate(TASKS[:n],1):
            job=run/'jobs'/f'tune-{i:02d}-{t}'; args=['--job-dir',str(job)]
            if previous: args+=['--input','parent-'+TASKS[i-2]+'='+str(previous/'artifacts/adapter.json')]
            self.assertEqual(b.fake_job(args),0); jobs.append(job); previous=job
        return jobs
    def test_default_chain_and_exact_parent_weights_policy(self):
        _,run,_=self.prepare({'tune':{'run':{'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}})
        s=self.execute(run); self.assertEqual(s['status'],'succeeded'); self.assertEqual(s['task_order'],list(TASKS)); self.assertEqual(self.peak(run),1)
        self.assertEqual(len({r['adapter']['path'] for r in s['jobs']}),6)
        launches=read_json(Path(s['invocation_dir'])/'execution.json')['launches']
        for i,row in enumerate(s['jobs']):
            d=run/'jobs'/row['job_id']; job,_=load_job(d); cmd=read_json(d/'execution.json')['command']; out=read_json(d/'artifacts/adapter.json')
            self.assertEqual(job['training_state_policy'],ex.CONTINUE if i else ex.FRESH); self.assertFalse(job['resume_training_state']); self.assertNotIn('--resume_checkpoint_path',cmd)
            self.assertEqual(out['metadata']['lineage'],job['lineage']); self.assertEqual(row['adapter']['slot'],TASKS[i])
            if i:
                parent=s['jobs'][i-1]; expected=read_json(parent['adapter']['manifest'])
                self.assertEqual([v for v in job['inputs'].values() if v['kind']=='adapter'],[expected]); self.assertIn('--adapter_name_or_path',cmd)
                self.assertEqual(launches[i]['command'][-2:],['--input','parent-'+TASKS[i-1]+'='+parent['adapter']['manifest']])
                self.assertEqual(out['metadata']['inherited_inputs'],{'parent-'+TASKS[i-1]:expected['fingerprint']})
                self.assertGreaterEqual(self.events(run)[row['job_id']]['started'],self.events(run)[parent['job_id']]['finished'])
            else: self.assertNotIn('--adapter_name_or_path',cmd); self.assertIsNone(row['parent_adapter'])
    def test_custom_reverse_order(self):
        order=list(reversed(TASKS)); _,run,_=self.prepare({'experiment':{'task_order':order}}); s=self.execute(run)
        self.assertEqual(s['status'],'succeeded'); self.assertEqual(s['task_order'],order)
        for i,row in enumerate(s['jobs']): self.assertEqual(row['depends_on'],[s['jobs'][i-1]['job_id']] if i else [])
    def test_first_failure_fail_fast(self):
        _,run,_=self.prepare(); s=self.execute(run,{'tune-01-motion':'failure'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),1); self.assertEqual([r['status'] for r in s['jobs']],['failed']+['cancelled']*5)
    def test_mid_failure_without_fail_fast(self):
        _,run,_=self.prepare({'tune':{'run':{'fail_fast':False}}}); s=self.execute(run,{'tune-02-hypo':'failure'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),2); self.assertEqual([r['status'] for r in s['jobs']],['succeeded','failed']+['cancelled']*4)
        self.assertTrue(all(r['blocked_by'] for r in s['jobs'][2:]))
    def test_corrupt_parent_blocks_without_fail_fast(self):
        _,run,_=self.prepare({'tune':{'run':{'fail_fast':False}}}); s=self.execute(run,{'tune-01-motion':'corrupt'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),1); self.assertIn('adapter audit failed',str(s['details'])); self.assertTrue(s['jobs'][1]['blocked_by'])
    def test_cleanup_failure_blocks(self):
        _,run,_=self.prepare({'tune':{'run':{'fail_fast':False}}}); s=self.execute(run,{'tune-01-motion':'cleanup'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),1); self.assertIn('cleanup',str(s['details']))
    def test_abrupt_exit_blocks(self):
        _,run,_=self.prepare({'tune':{'run':{'fail_fast':False}}}); s=self.execute(run,{'tune-01-motion':'abrupt'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),1)
    def test_spawn_failure_blocks(self):
        _,run,_=self.prepare({'tune':{'run':{'fail_fast':False}}})
        with patch.object(orch.subprocess,'Popen',side_effect=OSError('simulated spawn error')): s=orch.execute_tune_run(run)
        self.assertEqual(s['status'],'failed'); self.assertFalse(self.events(run)); self.assertTrue(all(x['status']=='cancelled' for x in s['jobs'][1:]))
    def test_prefix_takeover_preserves_existing_receipts(self):
        _,run,_=self.prepare(); prefix=self.prefix(run,2); before=[(p/'execution.json').read_bytes() for p in prefix]; s=self.execute(run)
        self.assertEqual(s['status'],'succeeded'); self.assertEqual(before,[(p/'execution.json').read_bytes() for p in prefix]); self.assertEqual(len(read_json(Path(s['invocation_dir'])/'execution.json')['launches']),4)
    def test_corrupt_existing_prefix_rejected(self):
        _,run,_=self.prepare(); parent=self.prefix(run,1)[0]; (Path(load_job(parent)[0]['output_dir'])/'adapter/adapter_model.safetensors').write_bytes(b'changed'); self.reject(run)
    def test_missing_bound_parent_rejected(self):
        _,run,_=self.prepare(); child=self.prefix(run,2)[1]; p=child/'job.json'; r=read_json(p); r['inputs']={k:v for k,v in r['inputs'].items() if v['kind']!='adapter'}; write_json_atomic(p,r); self.reject(run)
    def test_wrong_training_policy_rejected(self):
        _,run,_=self.prepare(); p=run/'jobs/tune-02-hypo/job.json'; r=read_json(p); r['training_state_policy']=ex.FRESH; write_json_atomic(p,r); self.reject(run)
    def test_plan_reordering_and_parent_tampering_rejected(self):
        for field in ('order','lineage'):
            _,run,_=self.prepare(); p=run/'plan.json'; r=read_json(p)
            if field=='order': r['jobs']=list(reversed(r['jobs']))
            else: r['jobs'][1]['lineage']['parent_job']='tune-06-geo'
            write_json_atomic(p,r); self.reject(run)
    def test_parent_must_have_passed_output_audit(self):
        _,run,_=self.prepare(); *_,jobs=orch._context(run); child,_=load_job(run/'jobs/tune-02-hypo')
        with self.assertRaisesRegex(ValueError,'not passed its output audit'): orch._parent_adapter(child,jobs)
    def test_changed_parent_registration_rejected(self):
        _,run,_=self.prepare(); self.prefix(run,1); *_,jobs=orch._context(run); child,_=load_job(run/'jobs/tune-02-hypo'); jobs['tune-01-motion']['adapter']['slot']='geo'
        with self.assertRaisesRegex(ValueError,'registration changed'): orch._parent_adapter(child,jobs)
    def test_exact_bound_parent_manifest_checked(self):
        _,run,_=self.prepare(); self.prefix(run,2); *_,jobs=orch._context(run); child,_=load_job(run/'jobs/tune-02-hypo'); child=deepcopy(child); role=next(k for k,v in child['inputs'].items() if v['kind']=='adapter'); child['inputs'][role]=read_json(run/'jobs/tune-02-hypo/artifacts/adapter.json')
        with self.assertRaisesRegex(ValueError,'differs from the registered parent'): orch._parent_adapter(child,jobs,require_bound=True)
    def test_cli_cl_tune_only_preserves_other_stages(self):
        p=self.config('cl'); q=subprocess.run([sys.executable,'-B',str(b.SELF),'--fake-high','--config',str(p),'--experiment-id','cli-cl'],capture_output=True,text=True); self.assertEqual(q.returncode,0,q.stderr)
        exp=Path(load_config(p)['config']['experiment']['records_root'])/'cli-cl'; rec,_=b.load_experiment(exp); self.assertEqual({x['stage']:record_status(exp/x['path'])['status'] for x in rec['stage_runs']},{'tune':'succeeded','infer':'planned','eval':'planned'})
    def test_restore_policy_is_distinct_from_cross_task_inheritance(self):
        _,run,_=self.prepare(); j,_=load_job(run/'jobs/tune-02-hypo'); j=deepcopy(j); j.update(resume_training_state=True,training_state_policy=ex.RESTORE); j['inputs']={'parent':{'kind':'adapter'},'resume':{'kind':'checkpoint'}}; orch._training_policy(j); j['training_state_policy']=ex.CONTINUE
        with self.assertRaises(ValueError): orch._training_policy(j)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--result-dir',required=True,type=Path); a=p.parse_args(); b.RESULT=a.result_dir.resolve(strict=True)
    if not b.RESULT.is_relative_to((b.REPO/'smoke/results').resolve(strict=True)): p.error('results outside smoke')
    if (b.RESULT/'tests.json').exists(): p.error('refusing existing evidence')
    os.environ['CUDA_VISIBLE_DEVICES']=''; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CLTests)); write_json_atomic(b.RESULT/'tests.json',{'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'successful':r.wasSuccessful()}); return 0 if r.wasSuccessful() else 1
if __name__=='__main__': raise SystemExit(main())

