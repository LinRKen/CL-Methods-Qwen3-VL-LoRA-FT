"""CPU acceptance for 36-job transfer tune plans and executors, without GPUs."""
import argparse,json,os,subprocess,sys,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import test_tune_orchestrator as b
from lib import tune_executor as ex,tune_orchestrator as orch
from lib.common import TASKS,read_json,write_json_atomic
from lib.records import load_job,record_status
from lib.config import load_config

class TransferTests(unittest.TestCase):
    setUp=b.T.setUp
    config=b.T.config
    execute=b.T.execute
    events=b.T.events
    peak=b.T.peak
    reject=b.T.reject
    def prepare(self,setting='transfer_round1',extra=None):
        return b.T.prepare(self,setting,extra)
    def one(self,run,job_id,parent=None):
        d=run/'jobs'/job_id; args=['--job-dir',str(d)]
        if parent: args+=['--input','parent-'+parent.removeprefix('tune-round1-')+'='+str(run/'jobs'/parent/'artifacts/adapter.json')]
        self.assertEqual(b.fake_job(args),0); return d
    def check_success(self,exp,run,s):
        self.assertEqual(s['status'],'succeeded'); self.assertEqual(len(s['jobs']),36)
        self.assertEqual(len({x['adapter']['path'] for x in s['jobs']}),36)
        self.assertEqual(record_status(exp)['status'],'active')
        self.assertEqual({x['stage']:record_status(exp/x['path'])['status'] for x in record_status(exp)['stage_runs']},{'tune':'succeeded','infer':'planned','eval':'planned'})
        rows={x['job_id']:x for x in s['jobs']}; events=self.events(run)
        for job_id,row in rows.items():
            d=run/'jobs'/job_id; job,_=load_job(d); rec=read_json(d/'execution.json'); cmd=rec['command']; out=read_json(d/'artifacts/adapter.json')
            self.assertEqual(job['lineage'],out['metadata']['lineage']); self.assertEqual(row['adapter']['slot'],job['output_spec']['slot'])
            self.assertFalse(job['resume_training_state']); self.assertNotIn('--resume_checkpoint_path',cmd)
            parent=job['lineage'].get('parent_job')
            self.assertEqual(job['training_state_policy'],ex.CONTINUE if parent else ex.FRESH)
            if parent:
                expected=read_json(run/'jobs'/parent/'artifacts/adapter.json'); source=job['lineage']['source_task']
                self.assertEqual(job['depends_on'],['tune-round1-'+source]); self.assertNotEqual(source,job['task'])
                self.assertEqual([v for v in job['inputs'].values() if v['kind']=='adapter'],[expected])
                self.assertEqual(cmd[cmd.index('--adapter_name_or_path')+1],expected['path'])
                self.assertEqual(out['metadata']['inherited_inputs'],{'parent-'+source:expected['fingerprint']})
                self.assertEqual(row['parent_adapter']['producer'],expected['producer'])
                self.assertGreaterEqual(events[job_id]['started'],events[parent]['finished'])
            else:
                self.assertEqual(job['depends_on'],[]); self.assertEqual(job['requires'],[]); self.assertIsNone(row['parent_adapter'])
                self.assertNotIn('--adapter_name_or_path',cmd); self.assertEqual(out['metadata']['inherited_inputs'],{})
    def test_both_full_topologies_and_eval_task_distinction(self):
        plans=[]
        for setting in ('transfer_round1','transfer_round2'):
            snap=load_config(self.config(setting)); tune=b.pipeline.build_stage_plan(snap,'tune')['jobs']; plans.append(tune)
            infer=b.pipeline.build_stage_plan(snap,'infer')['jobs']; eva=b.pipeline.build_stage_plan(snap,'eval')['jobs']
            self.assertEqual(len(tune),36); self.assertEqual(len(infer),30); self.assertEqual(len(eva),1); self.assertEqual(len(eva[0]['lineage']['scoring_pairs']),30)
            self.assertEqual([x['task'] for x in tune[:6]],list(TASKS)); self.assertEqual(len({x['output']['slot'] for x in tune}),36)
            self.assertEqual({(x['lineage']['source_task'],x['task']) for x in tune[6:]},{(s,t) for s in TASKS for t in TASKS if s!=t})
            for x in infer:
                task=x['lineage']['source_task' if setting=='transfer_round1' else 'second_round_task']
                self.assertEqual(x['tasks'],[task]); self.assertEqual(x['requires'][0]['job_id'],x['lineage']['adapter_ref']); self.assertEqual(x['lineage']['model_loads_per_job'],1)
                pair=next(z for z in eva[0]['lineage']['scoring_pairs'] if z['prediction_job']==x['job_id']); self.assertEqual(pair['task'],task)
        # Paths differ per YAML only through unchanged experiment roots.
        self.assertEqual(plans[0],plans[1])
    def test_round1_all36_exact_parent_and_sibling_independence(self):
        exp,run,_=self.prepare(); self.check_success(exp,run,self.execute(run)); self.assertEqual(self.peak(run),1)
    def test_round2_all36_parallel_resources_and_exact_parent(self):
        exp,run,_=self.prepare('transfer_round2',{'tune':{'run':{'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}}); self.check_success(exp,run,self.execute(run)); self.assertEqual(self.peak(run),2)
        jobs=list(self.events(run).items()); receipts={j:read_json(run/'jobs'/j/'execution.json') for j,_ in jobs}
        for i,(a,ea) in enumerate(jobs):
            for z,ez in jobs[i+1:]:
                if max(ea['started'],ez['started'])<min(ea['finished'],ez['finished']): self.assertTrue(set(receipts[a]['gpu_identity']['physical_ids']).isdisjoint(receipts[z]['gpu_identity']['physical_ids']))
    def test_parent_failure_only_blocks_its_five_children(self):
        _,run,_=self.prepare(extra={'tune':{'run':{'fail_fast':False,'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}}); s=self.execute(run,{'tune-round1-motion':'failure'}); rows={x['job_id']:x for x in s['jobs']}
        self.assertEqual(s['status'],'failed'); self.assertEqual(sum(x['status']=='succeeded' for x in rows.values()),30); self.assertEqual(len(self.events(run)),31)
        for target in TASKS[1:]:
            j='tune-round2-motion-to-'+target; self.assertEqual(rows[j]['status'],'cancelled'); self.assertEqual(rows[j]['blocked_by'],['tune-round1-motion']); self.assertNotIn(j,self.events(run))
    def test_second_round_failure_does_not_block_siblings(self):
        _,run,_=self.prepare(extra={'tune':{'run':{'fail_fast':False,'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}}); s=self.execute(run,{'tune-round2-motion-to-hypo':'failure'})
        self.assertEqual(s['status'],'failed'); self.assertEqual(sum(x['status']=='succeeded' for x in s['jobs']),35); self.assertEqual(len(self.events(run)),36)
    def test_fail_fast_stops_after_first_failure(self):
        _,run,_=self.prepare(); s=self.execute(run,{'tune-round1-motion':'failure'}); self.assertEqual(len(self.events(run)),1); self.assertEqual([x['status'] for x in s['jobs']],['failed']+['cancelled']*35)
    def test_corrupt_parent_blocked_before_child_spawn(self):
        _,run,_=self.prepare(); s=self.execute(run,{'tune-round1-motion':'corrupt'}); self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(run)),1); self.assertIn('adapter audit failed',str(s['details']))
    def test_existing_partial_branch_takeover_never_relaunches_success(self):
        exp,run,_=self.prepare(extra={'tune':{'run':{'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}}); parent='tune-round1-motion'; a=self.one(run,parent); z=self.one(run,'tune-round2-motion-to-hypo',parent); old=[(d/'execution.json').read_bytes() for d in (a,z)]
        s=self.execute(run); self.check_success(exp,run,s); self.assertEqual(old,[(d/'execution.json').read_bytes() for d in (a,z)]); self.assertEqual(len(read_json(Path(s['invocation_dir'])/'execution.json')['launches']),34)
    def test_source_does_not_wait_for_unrelated_round1_parent(self):
        _,run,_=self.prepare(extra={'tune':{'run':{'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}})
        for t in TASKS[:-1]: self.one(run,'tune-round1-'+t)
        def delayed(command,**kw):
            if command[command.index('--job-dir')+1].endswith('tune-round1-geo'):
                return b.REAL_POPEN([sys.executable,'-B','-c','import sys,time; time.sleep(12); import os; os.execv(sys.executable,[sys.executable]+sys.argv[1:])',str(b.SELF),'--fake-job',*command[2:]],**kw)
            return b.dispatch(command,**kw)
        with patch.object(orch.subprocess,'Popen',side_effect=delayed): s=orch.execute_tune_run(run)
        self.assertEqual(s['status'],'succeeded'); e=self.events(run); self.assertLess(e['tune-round2-motion-to-hypo']['started'],e['tune-round1-geo']['started'])
    def test_invalid_topology_rejected_before_launch(self):
        for mode in ('parent','order','source','self_target','requires','slot','annotation'):
            with self.subTest(mode=mode):
                _,run,_=self.prepare(); p=run/'plan.json'; r=read_json(p); j=r['jobs'][6]
                if mode=='parent': j['depends_on']=['tune-round1-hypo']
                elif mode=='order': r['jobs'][6:8]=reversed(r['jobs'][6:8])
                elif mode=='source': j['lineage']['source_task']='geo'
                elif mode=='self_target': j['task']='motion'
                elif mode=='requires': j['requires'][0]['slot']='round1-hypo'
                elif mode=='slot': j['output']['slot']='round1-motion'
                else: j['input_specs'][-1]['metadata']['split']='test'
                write_json_atomic(p,r); self.reject(run)
    def test_missing_duplicate_and_self_jobs_rejected(self):
        for mode in ('missing','duplicate','self'):
            _,run,_=self.prepare(); p=run/'plan.json'; r=read_json(p)
            if mode=='missing': r['jobs'].pop()
            elif mode=='duplicate': r['jobs'].append(deepcopy(r['jobs'][6]))
            else: r['jobs'][6]['job_id']='tune-round2-motion-to-motion'
            write_json_atomic(p,r); self.reject(run)
    def test_training_policy_tampering_rejected(self):
        _,run,_=self.prepare(); p=run/'jobs/tune-round2-motion-to-hypo/job.json'; r=read_json(p); r['training_state_policy']=ex.FRESH; write_json_atomic(p,r); self.reject(run)
    def test_corrupt_existing_parent_rejected_without_launch(self):
        _,run,_=self.prepare(); d=self.one(run,'tune-round1-motion'); (Path(load_job(d)[0]['output_dir'])/'adapter/adapter_model.safetensors').write_bytes(b'changed'); self.reject(run)
    def test_wrong_exact_bound_parent_is_rejected(self):
        _,run,_=self.prepare(); self.one(run,'tune-round1-motion'); self.one(run,'tune-round1-hypo'); *_,jobs=orch._context(run); child,_=load_job(run/'jobs/tune-round2-motion-to-hypo'); child=deepcopy(child); child['inputs']={'parent-motion':read_json(run/'jobs/tune-round1-hypo/artifacts/adapter.json')}
        with self.assertRaisesRegex(ValueError,'differs from the registered parent'): orch._parent_adapter(child,jobs,require_bound=True)
    def test_restore_policy_is_separate_from_cross_task_inheritance(self):
        _,run,_=self.prepare(); j,_=load_job(run/'jobs/tune-round2-motion-to-hypo'); j=deepcopy(j); j.update(resume_training_state=True,training_state_policy=ex.RESTORE); j['inputs']={'parent':{'kind':'adapter'},'checkpoint':{'kind':'checkpoint'}}; orch._training_policy(j); j['training_state_policy']=ex.CONTINUE
        with self.assertRaises(ValueError): orch._training_policy(j)
    def test_transfer_cli_executes_only_tune_from_frozen_snapshot(self):
        exp,run,p=self.prepare('transfer_round2'); p.write_text('invalid: ['); q=subprocess.run([sys.executable,'-B',str(b.SELF),'--fake-high','--run-dir',str(run)],capture_output=True,text=True); self.assertEqual(q.returncode,0,q.stderr); self.assertEqual(record_status(run)['status'],'succeeded'); self.assertEqual([record_status(exp/x['path'])['status'] for x in record_status(exp)['stage_runs'] if x['stage']!='tune'],['planned','planned'])
    def test_cpu_never_imports_model_libraries(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','peft','wandb','deepspeed','vllm'})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--result-dir',required=True,type=Path); a=p.parse_args(); b.RESULT=a.result_dir.resolve(strict=True)
    if not b.RESULT.is_relative_to((b.REPO/'smoke/results').resolve(strict=True)): p.error('results outside smoke')
    if (b.RESULT/'tests.json').exists() or (b.RESULT/'cases').exists(): p.error('refusing existing evidence')
    os.environ['CUDA_VISIBLE_DEVICES']=''; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransferTests)); write_json_atomic(b.RESULT/'tests.json',{'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'successful':r.wasSuccessful()}); return 0 if r.wasSuccessful() else 1
if __name__=='__main__': raise SystemExit(main())

