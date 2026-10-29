'''CPU acceptance for high-level infer; all model payloads are explicit fakes.'''
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch
LAUNCH_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
SELF=Path(__file__).resolve()
REPO=SELF.parents[2]
sys.path.insert(0,str(REPO/'scripts/experiments'))
sys.path.insert(0,str(SELF.parent))
_saved_argv=sys.argv[:]; sys.argv=[str(SELF)]
import test_infer_executor as fixture
sys.argv=_saved_argv
import pipeline
import run_infer
import yaml
from lib import infer_executor as ex, infer_worker as worker, infer_orchestrator as orch
from lib.common import TASKS,read_json,write_json_atomic,fingerprint
from lib.config import deep_merge
from lib.config import load_config
from lib.records import load_job,load_experiment,load_stage_run,record_status,bind_inputs,update_status,register_artifact
from lib.artifacts import make_manifest,check_manifest
REAL_POPEN=subprocess.Popen
REAL_EXCLUSIVE=ex._exclusive
RESULT=None

def identity(group,env):
    return {'physical_ids':list(group),'uuids':['CPU-FAKE-'+str(i) for i in group],'pci_bus_ids':['00000000:0'+str(i+2)+':00.0' for i in group]}

def fake_worker(request):
    req=read_json(request); original=worker._request
    def checked(path):
        ident=req['gpu_identity']
        rows=chr(10).join(str(i)+', '+u+', '+p for i,u,p in zip(ident['physical_ids'],ident['uuids'],ident['pci_bus_ids']))+chr(10)
        with patch.object(worker.subprocess,'run',return_value=types.SimpleNamespace(stdout=rows)):
            return original(path)
    with patch.object(worker,'_request',side_effect=checked),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':LAUNCH_CUDA or ''}):
        return fixture.fake_worker(request)

def fake_job(argv):
    parser=argparse.ArgumentParser(); parser.add_argument('--job-dir',required=True,type=Path); parser.add_argument('--gpu-group-index',type=int,default=0); parser.add_argument('--input',action='append',default=[]); a=parser.parse_args(argv)
    directory=a.job_dir
    mode=json.loads(os.environ.get('SMOKE_JOB_MODES','{}')).get(directory.name,'success')
    event={'pid':os.getpid(),'started':time.monotonic(),'mode':mode,'launch_cuda':LAUNCH_CUDA}
    write_json_atomic(directory/'cpu-event.json',event)
    lock_root=Path(load_job(directory)[1]['config']['experiment']['records_root']).parent/'cpu-resource-locks'
    def lock(path,**kw):
        if Path(path).parent==Path('/tmp/qwen3vl-lora-tune-locks'):path=lock_root/Path(path).name
        return REAL_EXCLUSIVE(path,**kw)
    def spawn(command,**kw):
        if mode=='abrupt':os._exit(19)
        time.sleep(2 if mode=='slow' else 0.05)
        env={**kw['env'],'SMOKE_INFER_MODE':mode if mode in ('failure','partial','badgt','shutdown','empty') else 'success'}
        return REAL_POPEN([sys.executable,'-B',str(SELF),'--fake-worker',command[-1]],**{**kw,'env':env})
    try:
        manifests={k:read_json(v) for k,v in (x.split('=',1) for x in a.input)}
        with patch.object(ex,'_exclusive',side_effect=lock),patch.object(ex,'_gpu_identity',side_effect=identity),patch.object(ex.subprocess,'Popen',side_effect=spawn):
            ex.execute_infer_job(directory,gpu_group_index=a.gpu_group_index,input_manifests=manifests)
        job,_=load_job(directory); output=Path(job['output_dir'])
        if mode=='corrupt':
            path=output/'predictions'/(job['tasks'][0]+'.jsonl'); path.write_bytes(b'corrupt')
        elif mode in ('receipt','cleanup'):
            receipt=read_json(directory/'execution.json')
            if mode=='receipt':receipt['config_fingerprint']='tampered'
            else:receipt['cleanup_completed']=False; receipt['cleanup_error']='smoke intentional cleanup failure'
            write_json_atomic(directory/'execution.json',receipt)
        elif mode=='request':
            req=read_json(directory/'inference.request.json'); req['params']['max_new_tokens']+=1
            write_json_atomic(directory/'inference.request.json',req)
        return 0
    except BaseException as error:
        event['error']=str(error)
        return 17
    finally:
        event.update(finished=time.monotonic(),status=record_status(directory)['status'])
        write_json_atomic(directory/'cpu-event.json',event)

def dispatch(command,**kw):
    return REAL_POPEN([command[0],'-B',str(SELF),'--fake-job',*command[2:]],**kw)

def fake_high(argv):
    with patch.object(orch.subprocess,'Popen',side_effect=dispatch):return run_infer.main(argv)

def adopt_cpu(directory,parents):
    job,snap=load_job(directory); cfg=snap['config']; params=cfg['tune']['params']
    inputs={s['role']:make_manifest(s['kind'],s['path'],source=s['source'],metadata=s['metadata']) for s in job['input_specs']}
    if job['depends_on']:inputs['adapter']=parents[job['depends_on'][0]]
    bind_inputs(directory,inputs); update_status(directory,'running','Explicit CPU fake adapter fixture; no training')
    out=Path(job['output_dir'])/'adapter'; out.mkdir(parents=True)
    write_json_atomic(out/'adapter_config.json',{'peft_type':'LORA','task_type':'CAUSAL_LM','r':params['lora_r'],'lora_alpha':params['lora_alpha'],'lora_dropout':params['lora_dropout'],'target_modules':['q_proj','k_proj','v_proj','o_proj'],'exclude_modules':'.*visual[.].*','bias':'none','base_model_name_or_path':cfg['experiment']['base_model']})
    (out/'adapter_model.safetensors').write_bytes(('CPU FAKE '+job['job_id']).encode())
    meta={'slot':job['output_spec']['slot'],'training_task':job['task'],'lineage':job['lineage'],'training_state_policy':job['training_state_policy'],'config_fingerprint':job['config_fingerprint'],'base_model_fingerprint':inputs['base-model']['fingerprint'],'input_fingerprints':{k:v['fingerprint'] for k,v in inputs.items()},'inherited_inputs':{k:v['fingerprint'] for k,v in inputs.items() if v['kind'] in ('adapter','checkpoint')},'smoke_fixture':True,'training_executed':False}
    manifest=make_manifest('adapter',out,producer={k:job[k] for k in ('experiment_id','stage','run_id','job_id')},metadata=meta)
    register_artifact(directory,'adapter',manifest); update_status(directory,'succeeded','CPU fixture only')
    parents[job['job_id']]=manifest

class T(unittest.TestCase):
    def setUp(self):
        self.case=RESULT/'cases'/self._testMethodName; self.case.mkdir(parents=True); self.n=0
        base=self.case/'base'; base.mkdir(); (base/'config.json').write_text('{}')
        data=self.case/'dataset'
        for task in (*TASKS,'whole'):
            (data/task).mkdir(parents=True)
            samples=[{'conversations':[{'from':'human','value':'question '+task},{'from':'gpt','value':'answer '+task}],'_meta':{'id':task+str(i),'sample_index':i+13,'task_type':'OPEN','metric_type':'ACC','nested':{'retain':True}}} for i in range(2)]
            for split in ('train','test'):write_json_atomic(data/task/(split+'.json'),samples)
        self.doc={'experiment':{'name':'infer-orchestrator-smoke','setting':'zero_shot','enabled_stages':['infer','eval'],'base_model':str(base),'processor_path':str(base),'dataset_root':str(data),**{k:str(self.case/k) for k in ('records_root','checkpoints_root','infer_root','eval_root')},'logs_root':str(self.case/'logs_root')},'infer':{'run':{'python':sys.executable,'gpu_groups':[[1]],'max_parallel_jobs':1},'params':{'batch_size':2,'max_new_tokens':8,'max_model_len':8192}},'tune':{'params':{'lora_r':8,'lora_alpha':16}}}
    def prepare(self,setting='zero_shot',extra=None,adopt=True):
        self.n+=1; doc=deep_merge(self.doc,{'experiment':{'setting':setting,'enabled_stages':['infer','eval'] if setting=='zero_shot' else ['tune','infer','eval']}})
        if extra:doc=deep_merge(doc,extra)
        path=self.case/('setting-'+str(self.n)+'.yaml'); path.write_text(yaml.safe_dump(doc)); snap=load_config(path); exp=pipeline.prepare_records(snap,'case-'+str(self.n))
        runs={x['stage']:exp/x['path'] for x in record_status(exp)['stage_runs']}
        if 'tune' in runs and adopt:
            update_status(runs['tune'],'running','CPU adoption only'); parents={}
            for j in record_status(runs['tune'])['jobs']:adopt_cpu(runs['tune']/j['path'],parents)
            update_status(runs['tune'],'succeeded','CPU fake producer fixtures complete; training not executed')
        return exp,runs,path
    def invocation_root(self,run):
        record=read_json(run/'run.json');snapshot=read_json(run/record['config_snapshot']);config=snapshot['config']['experiment']
        return Path(config.get('logs_root',config['records_root']))/record['experiment_id']/'stage-runs'/record['stage']/record['run_id']/'orchestrations'
    def assert_logged_result(self,run,result):
        folder=Path(result['invocation_dir']); self.assertTrue(folder.is_relative_to(self.invocation_root(run)))
        receipt=read_json(folder/'invocation.json'); self.assertEqual(receipt['status'],result['status'])
        self.assertTrue((folder/'orchestrator.log').is_file()); self.assertGreater((folder/'orchestrator.log').stat().st_size,0)
        if result['status']=='succeeded':self.assertFalse((folder/'failure.json').exists())
        else:
            failure=read_json(folder/'failure.json'); self.assertEqual(failure['status'],result['status']); self.assertTrue(failure.get('details') or failure.get('traceback'))
    def assert_new_failed_invocation(self,run,before):
        added=set(self.invocation_root(run).glob('*'))-before; self.assertEqual(len(added),1,[str(x) for x in added])
        folder=added.pop(); receipt=read_json(folder/'invocation.json'); failure=read_json(folder/'failure.json')
        self.assertEqual(receipt['status'],'failed'); self.assertTrue(failure.get('traceback') or failure.get('details')); self.assertTrue((folder/'orchestrator.log').is_file())
    def execute(self,run,modes=None):
        with patch.dict(os.environ,{'SMOKE_JOB_MODES':json.dumps(modes or {})}),patch.object(orch.subprocess,'Popen',side_effect=dispatch):result=orch.execute_infer_run(run)
        self.assert_logged_result(run,result); return result
    def events(self,run):return {p.parent.name:read_json(p) for p in (run/'jobs').glob('*/cpu-event.json')}
    def peak(self,run):
        points=[]
        for e in self.events(run).values():
            self.assertIn('finished',e); points.extend([(e['started'],1),(e['finished'],-1)])
        current=peak=0
        for _,v in sorted(points):current+=v; peak=max(peak,current)
        return peak
    def reject(self,run):
        before=set(self.invocation_root(run).glob('*'))
        with patch.object(orch.subprocess,'Popen') as spawn,self.assertRaises((ValueError,OSError,KeyError,TypeError)):orch.execute_infer_run(run)
        spawn.assert_not_called(); self.assert_new_failed_invocation(run,before)
    def first(self,run):return run/'jobs'/record_status(run)['jobs'][0]['job_id']
    def test_01_import_and_compile_cpu_only(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','vllm','peft','wandb','deepspeed'})
        for f in ('run_infer.py','lib/infer_orchestrator.py'):compile((REPO/'scripts/experiments'/f).read_text(),f,'exec')
    def test_02_all_settings_bind_exact_adapters_and_test_tasks(self):
        for setting,total in [('zero_shot',6),('multitask',6),('singletask',6),('cl',36),('transfer_round1',30),('transfer_round2',30)]:
            with self.subTest(setting=setting):
                exp,runs,_=self.prepare(setting); *_,jobs=orch._context(runs['infer'])
                self.assertEqual(sum(len(i['record']['tasks']) for i in jobs.values()),total)
                self.assertEqual(len(jobs),{'zero_shot':1,'multitask':1,'singletask':6,'cl':6,'transfer_round1':30,'transfer_round2':30}[setting])
                for item in jobs.values():
                    job=item['record']; tasks=job['tasks']
                    if setting=='zero_shot':self.assertIsNone(item['adapter']); continue
                    source=job['lineage']['adapter_ref']; manifest=read_json(runs['tune']/'jobs'/source/'artifacts/adapter.json')
                    self.assertEqual(item['adapter']['fingerprint'],manifest['fingerprint']); self.assertEqual(item['adapter']['producer']['experiment_id'],exp.name)
                    if setting in ('multitask','cl'):self.assertEqual(tasks,list(TASKS))
                    elif setting=='singletask':self.assertEqual(tasks,[load_job(runs['tune']/'jobs'/source)[0]['task']])
                    else:
                        lineage=load_job(runs['tune']/'jobs'/source)[0]['lineage']; self.assertEqual(tasks,[lineage['source_task' if setting=='transfer_round1' else 'train_task']])
    def test_03_custom_cl_order_keeps_each_adapter_six_test_tasks(self):
        order=list(reversed(TASKS)); _,runs,_=self.prepare('cl',{'experiment':{'task_order':order}}); *_,jobs=orch._context(runs['infer'])
        self.assertEqual([i['record']['lineage']['adapter_ref'] for i in jobs.values()],['tune-'+format(n,'02d')+'-'+t for n,t in enumerate(order,1)])
        self.assertTrue(all(i['record']['tasks']==list(TASKS) for i in jobs.values()))
    def test_04_zero_shot_executes_one_job_six_tasks_one_load(self):
        exp,runs,_=self.prepare(); s=self.execute(runs['infer']); self.assertEqual(s['status'],'succeeded'); self.assertEqual(len(s['jobs']),1)
        j=s['jobs'][0]; events=read_json(Path(j['output_dir'])/'fake-events.json'); self.assertEqual(events.count('load-engine'),1); self.assertEqual(events.count('shutdown'),1)
        self.assertEqual(len([e for e in events if e.startswith('infer:')]),6); self.assertEqual(len(j['artifacts']),7)
        self.assertEqual(read_json(runs['infer']/'infer-summary.json'),s); self.assertEqual(record_status(exp)['status'],'active'); self.assertEqual(record_status(runs['eval'])['status'],'planned')
    def test_05_multitask_one_adapter_all_tasks(self):
        _,r,_=self.prepare('multitask'); s=self.execute(r['infer']); self.assertEqual(s['status'],'succeeded'); self.assertEqual(s['jobs'][0]['adapter']['slot'],'whole'); self.assertEqual(len(s['jobs'][0]['artifacts']),7)
    def test_06_singletask_six_distinct_adapters(self):
        _,r,_=self.prepare('singletask'); s=self.execute(r['infer']); self.assertEqual(s['status'],'succeeded'); self.assertEqual([j['tasks'] for j in s['jobs']],[[t] for t in TASKS]); self.assertEqual(len({j['adapter']['fingerprint'] for j in s['jobs']}),6); self.assertEqual(self.peak(r['infer']),1)
    def test_07_transfer_round2_full_thirty_job_execution(self):
        _,r,_=self.prepare('transfer_round2',{'infer':{'run':{'max_parallel_jobs':2,'gpu_groups':[[1],[2]]}}}); s=self.execute(r['infer']); self.assertEqual(s['status'],'succeeded'); self.assertEqual(len(s['jobs']),30); self.assertEqual(sum(len(j['tasks']) for j in s['jobs']),30)
        self.assertEqual(len({j['adapter']['fingerprint'] for j in s['jobs']}),30)
        for j in s['jobs']:self.assertEqual(read_json(Path(j['output_dir'])/'fake-events.json').count('load-engine'),1)
    def test_08_infer_only_completes_experiment(self):
        exp,r,_=self.prepare(extra={'experiment':{'enabled_stages':['infer']}}); self.execute(r['infer']); self.assertEqual(record_status(exp)['status'],'succeeded')
    def test_09_disjoint_gpu_groups_allow_parallel(self):
        _,r,_=self.prepare('singletask',{'infer':{'run':{'max_parallel_jobs':2,'gpu_groups':[[1],[2]]}}}); self.assertEqual(self.execute(r['infer'])['status'],'succeeded'); self.assertEqual(self.peak(r['infer']),2)
    def test_10_same_job_groups_serialize(self):
        jobs={'infer-'+t:{'run':{'gpu_groups':[[1]],'max_parallel_jobs':1}} for t in TASKS}
        _,r,_=self.prepare('singletask',{'infer':{'run':{'max_parallel_jobs':2,'gpu_groups':[[1],[2]]},'jobs':jobs}}); self.execute(r['infer']); self.assertEqual(self.peak(r['infer']),1)
    def test_11_job_override_reaches_request_and_assignment(self):
        _,r,_=self.prepare('singletask',{'infer':{'jobs':{'infer-hypo':{'run':{'gpu_groups':[[3]]},'params':{'batch_size':1,'max_new_tokens':19,'video_max_frames':6}}}}}); s=self.execute(r['infer']); self.assertEqual(s['status'],'succeeded'); j=next(x for x in s['jobs'] if x['job_id']=='infer-hypo'); req=read_json(Path(j['output_dir'])/'inference.request.json')
        self.assertEqual(j['assignment']['physical_gpu_ids'],[3]); self.assertEqual(req['params']['max_new_tokens'],19); self.assertEqual(req['params']['video_max_frames'],6); self.assertEqual(req['params']['batch_size'],1)
    def test_12_fail_fast_cancels_pending(self):
        _,r,_=self.prepare('singletask'); job=self.first(r['infer']).name; s=self.execute(r['infer'],{job:'failure'}); self.assertEqual(s['status'],'failed'); self.assertEqual([j['status'] for j in s['jobs']],['failed']+['cancelled']*5); self.assertEqual(len(self.events(r['infer'])),1)
    def test_13_continue_independent_jobs_without_fail_fast(self):
        _,r,_=self.prepare('singletask',{'infer':{'run':{'fail_fast':False}}}); s=self.execute(r['infer'],{self.first(r['infer']).name:'failure'}); self.assertEqual(s['status'],'failed'); self.assertEqual([j['status'] for j in s['jobs']],['failed']+['succeeded']*5)
    def test_14_cleanup_failure_stops_even_without_fail_fast(self):
        _,r,_=self.prepare('singletask',{'infer':{'run':{'fail_fast':False}}}); s=self.execute(r['infer'],{self.first(r['infer']).name:'cleanup'}); self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(r['infer'])),1); self.assertIn('cleanup',str(s['details']))
    def test_15_abrupt_exit_stops_even_without_fail_fast(self):
        _,r,_=self.prepare('singletask',{'infer':{'run':{'fail_fast':False}}}); s=self.execute(r['infer'],{self.first(r['infer']).name:'abrupt'}); self.assertEqual(s['status'],'failed'); self.assertEqual(len(self.events(r['infer'])),1)
    def test_16_corrupt_success_rejected_by_parent(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'corrupt'}); self.assertEqual(s['status'],'failed'); self.assertIn('prediction audit failed',str(s['details']))
    def test_17_bad_receipt_rejected_by_parent(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'receipt'}); self.assertEqual(s['status'],'failed'); self.assertIn('prediction audit failed',str(s['details']))
    def test_18_request_tamper_rejected_by_parent(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'request'}); self.assertEqual(s['status'],'failed'); self.assertIn('prediction audit failed',str(s['details']))
    def test_19_existing_verified_success_is_skipped(self):
        _,r,_=self.prepare('singletask'); *_,jobs=orch._context(r['infer']); first=next(iter(jobs.values())); directory=first['directory']; code=fake_job(['--job-dir',str(directory),'--input','adapter='+first['adapter']['manifest']]); self.assertEqual(code,0); before=(directory/'execution.json').read_bytes(); s=self.execute(r['infer']); self.assertEqual(s['status'],'succeeded'); self.assertEqual((directory/'execution.json').read_bytes(),before); self.assertIsNone(s['jobs'][0]['log']); self.assertEqual(len(read_json(Path(s['invocation_dir'])/'execution.json')['launches']),5)
    def test_20_existing_corrupt_success_rejected_before_launch(self):
        _,r,_=self.prepare(); directory=self.first(r['infer']); self.assertEqual(fake_job(['--job-dir',str(directory)]),0); job,_=load_job(directory); (Path(job['output_dir'])/'predictions/motion.jsonl').write_bytes(b'x'); self.reject(r['infer'])
    def test_21_terminal_stage_not_restarted(self):
        _,r,_=self.prepare(); self.execute(r['infer']); self.reject(r['infer'])
    def test_22_planned_execution_and_output_preserved(self):
        _,r,_=self.prepare(); (self.first(r['infer'])/'execution.json').write_text('keep'); self.reject(r['infer'])
        _,r,_=self.prepare(); job,_=load_job(self.first(r['infer'])); out=Path(job['output_dir']); out.mkdir(parents=True); (out/'sentinel').write_text('keep'); self.reject(r['infer']); self.assertEqual((out/'sentinel').read_text(),'keep')
    def test_23_running_failed_cancelled_jobs_not_retried(self):
        for state in ('running','failed','cancelled'):
            _,r,_=self.prepare(); path=self.first(r['infer'])/'job.json'; doc=read_json(path); doc['status']=state; write_json_atomic(path,doc); self.reject(r['infer'])
    def test_24_plan_and_job_tampering_rejected(self):
        _,r,_=self.prepare(); path=r['infer']/'plan.json'; doc=read_json(path); doc['jobs'][0]['tasks']=['geo']; write_json_atomic(path,doc); self.reject(r['infer'])
        _,r,_=self.prepare(); path=self.first(r['infer'])/'job.json'; doc=read_json(path); doc['tasks']=['geo']; write_json_atomic(path,doc); self.reject(r['infer'])
    def test_25_missing_successful_tune_run_rejected(self):
        _,r,_=self.prepare('multitask',adopt=False); self.reject(r['infer'])
    def test_26_duplicate_tune_run_rejected(self):
        exp,r,_=self.prepare('multitask'); path=exp/'experiment.json'; doc=read_json(path); doc['stage_runs'].append(deepcopy(next(x for x in doc['stage_runs'] if x['stage']=='tune'))); write_json_atomic(path,doc); self.reject(r['infer'])
    def test_27_source_adapter_payload_tamper_rejected(self):
        _,r,_=self.prepare('multitask'); directory=r['tune']/'jobs/tune-whole'; job,_=load_job(directory); (Path(job['output_dir'])/'adapter/adapter_model.safetensors').write_bytes(b'tamper'); self.reject(r['infer'])
    def test_28_source_lineage_tamper_rejected(self):
        _,r,_=self.prepare('cl'); source=next((r['tune']/'jobs').iterdir()); path=source/'job.json'; doc=read_json(path); doc['lineage']['parent_job']='wrong'; write_json_atomic(path,doc); self.reject(r['infer'])
    def test_29_foreign_experiment_adapter_rejected(self):
        _,r1,_=self.prepare('multitask'); _,r2,_=self.prepare('multitask'); source=r1['tune']/'jobs/tune-whole'; dest=r2['tune']/'jobs/tune-whole'; m=read_json(source/'artifacts/adapter.json'); write_json_atomic(dest/'artifacts/adapter.json',m); self.reject(r2['infer'])
    def test_30_duplicate_high_level_lock_rejected(self):
        _,r,_=self.prepare()
        with orch._run_lock(r['infer']),self.assertRaisesRegex(ValueError,'Another high-level'):orch.execute_infer_run(r['infer'])
    def test_31_invalid_gpu_resources_rejected(self):
        _,r,_=self.prepare(); cfg=load_job(self.first(r['infer']))[1]['config']
        for groups,tp in [([[0]],1),([[1,1]],2),([[1]],2),([[True]],1),([],1)]:
            bad=deepcopy(cfg); bad['infer']['run']['gpu_groups']=groups; bad['infer']['params']['tensor_parallel_size']=tp
            with self.assertRaises(ValueError):orch._resources(bad,'fake')
    def test_32_cli_frozen_snapshot_after_yaml_invalidated(self):
        _,r,path=self.prepare(); path.write_text('invalid: ['); q=subprocess.run([sys.executable,'-B',str(SELF),'--fake-high','--run-dir',str(r['infer'])],capture_output=True,text=True); self.assertEqual(q.returncode,0,q.stderr); self.assertEqual(record_status(r['infer'])['status'],'succeeded')
    def test_33_cli_only_accepts_absolute_frozen_run(self):
        _,r,_=self.prepare()
        for extra in (['--set','infer.params.batch_size=4'],['--config','x.yaml'],['--experiment-id','x']):
            q=subprocess.run([sys.executable,'-B',str(REPO/'scripts/experiments/run_infer.py'),'--run-dir',str(r['infer']),*extra],capture_output=True); self.assertEqual(q.returncode,2)
        q=subprocess.run([sys.executable,'-B',str(REPO/'scripts/experiments/run_infer.py'),'--run-dir','relative'],capture_output=True); self.assertEqual(q.returncode,1); self.assertIn(b'absolute',q.stderr)
    def test_34_cli_failed_run_returns_one(self):
        _,r,_=self.prepare(); q=subprocess.run([sys.executable,'-B',str(SELF),'--fake-high','--run-dir',str(r['infer'])],capture_output=True,env={**os.environ,'SMOKE_JOB_MODES':json.dumps({self.first(r['infer']).name:'failure'})}); self.assertEqual(q.returncode,1)
    def test_35_sigterm_cancels_and_joins_children(self):
        _,r,_=self.prepare('singletask'); directory=self.first(r['infer'])
        with (self.case/'signal.log').open('xb') as log:
            proc=REAL_POPEN([sys.executable,'-B',str(SELF),'--fake-high','--run-dir',str(r['infer'])],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'SMOKE_JOB_MODES':json.dumps({directory.name:'slow'})})
            try:
                end=time.monotonic()+20
                while not (directory/'execution.json').exists() or read_json(directory/'execution.json').get('status')!='running':
                    if time.monotonic()>end or proc.poll() is not None:self.fail('low-level executor did not start')
                    time.sleep(0.03)
                proc.send_signal(signal.SIGTERM); self.assertEqual(proc.wait(timeout=20),130)
            finally:
                if proc.poll() is None:proc.terminate(); proc.wait(timeout=20)
        self.assertEqual(record_status(r['infer'])['status'],'cancelled')
        for event in self.events(r['infer']).values():self.assertFalse(Path('/proc/'+str(event['pid'])).exists())
    def test_36_launch_failure_propagates_without_gpu_access(self):
        _,r,_=self.prepare()
        with patch.object(orch.subprocess,'Popen',side_effect=OSError('smoke launch failure')):s=orch.execute_infer_run(r['infer'])
        self.assertEqual(s['status'],'failed'); self.assertIn('smoke launch failure',str(s['details']))
    def test_37_shutdown_failure_fails_stage(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'shutdown'}); self.assertEqual(s['status'],'failed')
    def test_38_partial_output_fails_stage(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'partial'}); self.assertEqual(s['status'],'failed')
    def test_39_empty_predictions_preserved_not_hidden(self):
        _,r,_=self.prepare(); s=self.execute(r['infer'],{self.first(r['infer']).name:'empty'}); self.assertEqual(s['status'],'succeeded'); summary=read_json(Path(s['jobs'][0]['output_dir'])/'inference_result.json'); self.assertTrue(all(t['empty_predictions']==2 for t in summary['tasks']))
    def test_40_all_successful_children_resume_without_loading(self):
        _,r,_=self.prepare(); directory=self.first(r['infer']); self.assertEqual(fake_job(['--job-dir',str(directory)]),0); before=(directory/'execution.json').read_bytes()
        with patch.object(orch.subprocess,'Popen') as spawn:s=orch.execute_infer_run(r['infer'])
        spawn.assert_not_called(); self.assertEqual(s['status'],'succeeded'); self.assertEqual((directory/'execution.json').read_bytes(),before)

def main():
    global RESULT
    parser=argparse.ArgumentParser(); parser.add_argument('--result-root',required=True,type=Path); a=parser.parse_args(); RESULT=a.result_root.resolve(); RESULT.mkdir(parents=True,exist_ok=False)
    if not RESULT.is_relative_to((REPO/'smoke/results').resolve()):parser.error('results must be under smoke/results')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    write_json_atomic(RESULT/'summary.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'details':[{'test':str(t),'traceback':s} for t,s in result.failures+result.errors]})
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fake-worker':raise SystemExit(fake_worker(Path(sys.argv[2])))
    if len(sys.argv)>1 and sys.argv[1]=='--fake-job':raise SystemExit(fake_job(sys.argv[2:]))
    if len(sys.argv)>1 and sys.argv[1]=='--fake-high':raise SystemExit(fake_high(sys.argv[2:]))
    raise SystemExit(main())
