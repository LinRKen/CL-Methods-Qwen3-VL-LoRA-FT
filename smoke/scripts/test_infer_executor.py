'''CPU inference acceptance. Simulated payloads are never actual model results.'''
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/experiments'))
from lib.common import TASKS,read_json,write_json_atomic,fingerprint
from lib.config import load_config,resolve_run_config
from lib.records import create_experiment,create_stage_run,create_job,load_job,load_stage_run,bind_inputs,update_status,register_artifact
from lib.artifacts import make_manifest,check_manifest
from lib import infer_executor as ex,infer_worker as worker
import pipeline
import yaml

def producer_fixture(directory,adapter_path=None):
    '''Register an adopted fixture explicitly; no new training is claimed.'''
    job,snapshot=load_job(directory); cfg=snapshot['config']; params=cfg['tune']['params']
    inputs={s['role']:make_manifest(s['kind'],s['path'],source=s['source'],metadata=s['metadata']) for s in job['input_specs']}
    bind_inputs(directory,inputs); update_status(directory,'running','Smoke fixture adoption; training not executed')
    destination=Path(job['output_dir'])/'adapter'; destination.mkdir(parents=True)
    if adapter_path:
        for name in ('adapter_config.json','adapter_model.safetensors'):(destination/name).symlink_to(Path(adapter_path)/name)
    else:
        config={'peft_type':'LORA','task_type':'CAUSAL_LM','r':params['lora_r'],'lora_alpha':params['lora_alpha'],'lora_dropout':params['lora_dropout'],'target_modules':['q_proj','k_proj','v_proj','o_proj'],'exclude_modules':r'.*visual[.].*','bias':'none','base_model_name_or_path':cfg['experiment']['base_model']}
        write_json_atomic(destination/'adapter_config.json',config); (destination/'adapter_model.safetensors').write_bytes(b'FAKE CPU PAYLOAD')
    metadata={'slot':job['output_spec']['slot'],'training_task':job['task'],'lineage':job['lineage'],'training_state_policy':job['training_state_policy'],'config_fingerprint':job['config_fingerprint'],'base_model_fingerprint':inputs['base-model']['fingerprint'],'input_fingerprints':{k:v['fingerprint'] for k,v in inputs.items()},'smoke_fixture':True,'adopted_adapter':str(adapter_path) if adapter_path else None,'training_executed':False}
    manifest=make_manifest('adapter',destination,producer={k:job[k] for k in ('experiment_id','stage','run_id','job_id')},metadata=metadata)
    register_artifact(directory,'adapter',manifest); update_status(directory,'succeeded','Fixture registered; not a training result')
    return manifest

def fake_worker(request):
    req=read_json(request); out=Path(req['output_dir']); events=[]; mode=os.environ.get('SMOKE_INFER_MODE','success')
    proc=types.SimpleNamespace(image_processor=types.SimpleNamespace(size={},min_pixels=0,max_pixels=0),video_processor=types.SimpleNamespace(size={},min_pixels=0,max_pixels=0,min_frames=0,max_frames=0,fps=0,num_frames=None,do_sample_frames=False))
    legacy=types.ModuleType('qwenvl.infer.infer_cosine')
    legacy.validate_runtime=lambda args:events.append('validate')
    legacy.preflight_samples=lambda samples:{'image':max(len(s.get('image',[])) for s in samples),'video':max(len(s.get('video',[])) for s in samples)}
    legacy.load_processor=lambda args:proc
    legacy.load_adapter_config=lambda args:('fake',8) if args.adapter_path else None
    legacy.prepare_model_directory=lambda args,p,d:events.append('model-view:'+str(d))
    def engine(*args):
        events.append('load-engine'); return object(),object(),object() if req['adapter_path'] else None
    legacy.load_engine=engine
    def shutdown(engine):
        events.append('shutdown'); write_json_atomic(out/'fake-events.json',events)
        if mode=='shutdown':raise RuntimeError('intentional shutdown failure')
    legacy.shutdown_vllm_engine=shutdown
    def infer(engine,processor,samples,dataset,path,args,sampling,lora):
        events.append('infer:'+dataset)
        if mode=='failure':raise RuntimeError('intentional inference failure')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:
            for i,sample in enumerate(samples):
                if mode=='partial' and i==len(samples)-1:break
                meta=sample.get('_meta',{})
                row={'dataset':dataset,'inference_index':i,'sample_index':meta.get('sample_index',i),'id':meta.get('id'),'task_type':meta.get('task_type'),'metric_type':meta.get('metric_type'),'_meta':meta,'ground_truth':next(t['value'] for t in sample['conversations'] if t['from']=='gpt'),'prediction':'' if mode=='empty' else 'CPU FAKE','generation':{'max_new_tokens':args.max_new_tokens,'do_sample':args.do_sample,'temperature':args.temperature if args.do_sample else None,'top_p':args.top_p if args.do_sample else None}}
                if mode=='badgt':row['ground_truth']='WRONG'
                handle.write(json.dumps(row)+chr(10))
    legacy.run_inference=infer
    package=types.ModuleType('qwenvl'); package.__path__=[]
    sub=types.ModuleType('qwenvl.infer'); sub.__path__=[]; sub.infer_cosine=legacy
    with patch.dict(sys.modules,{'qwenvl':package,'qwenvl.infer':sub,'qwenvl.infer.infer_cosine':legacy}),patch.object(worker.subprocess,'run',return_value=types.SimpleNamespace(stdout='1, CPU-FAKE-1, 00000000:03:00.0'+chr(10))):
        return worker.main(['--request',str(request)])
if len(sys.argv)>1 and sys.argv[1]=='--fake-worker':raise SystemExit(fake_worker(Path(sys.argv[2])))
os.environ['CUDA_VISIBLE_DEVICES']=''
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','wandb','deepspeed','vllm'}:raise ImportError('Forbidden CPU model import: '+fullname)
sys.meta_path.insert(0,BlockModels())
REAL_POPEN=subprocess.Popen
REAL_EXCLUSIVE=ex._exclusive
RESULT=None

class InferTests(unittest.TestCase):
    def setUp(self):
        self.case=RESULT/'cases'/self._testMethodName; self.case.mkdir(parents=True); self.seq=0
        base=self.case/'base'; base.mkdir(); (base/'config.json').write_text('{}')
        data=self.case/'data'
        for task in (*TASKS,'whole'):
            (data/task).mkdir(parents=True)
            samples=[{'conversations':[{'from':'human','value':'question'},{'from':'gpt','value':'answer'}],'_meta':{'id':task+str(i),'sample_index':i+10,'task_type':'OPEN','metric_type':'ACC','extra':{'keep':True}}} for i in range(2)]
            for split in ('train','test'):write_json_atomic(data/task/(split+'.json'),samples)
        self.doc={'experiment':{'name':'infer-smoke','setting':'zero_shot','enabled_stages':['infer','eval'],'base_model':str(base),'processor_path':str(base),'dataset_root':str(data),**{k:str(self.case/k) for k in ('records_root','checkpoints_root','infer_root','eval_root')}},'infer':{'run':{'python':sys.executable,'gpu_groups':[[1]]},'params':{'batch_size':2,'max_new_tokens':8,'max_model_len':8192}},'tune':{'params':{'lora_r':8,'lora_alpha':16}}}
        self.yaml=self.case/'setting.yaml'; self.yaml.write_text(yaml.safe_dump(self.doc))
    def records(self,setting='zero_shot'):
        self.seq+=1; self.doc['experiment']['setting']=setting; self.doc['experiment']['enabled_stages']=['infer','eval'] if setting=='zero_shot' else ['tune','infer','eval']; self.yaml.write_text(yaml.safe_dump(self.doc)); snap=load_config(self.yaml); exp=create_experiment(snap,'exp-'+str(self.seq)); runs={}; jobs={}
        for stage in ('tune','infer'):
            if setting=='zero_shot' and stage=='tune':continue
            plan=pipeline.build_stage_plan(resolve_run_config(snap,stage),stage); runs[stage]=create_stage_run(exp,stage,run_id=stage+'-run',stage_plan=plan); jobs[stage]=[create_job(runs[stage],**item) for item in plan['jobs']]
        return exp,runs,jobs
    @contextmanager
    def locks(self,path,blocking=False):
        path=Path(path)
        if str(path).startswith('/tmp/qwen3vl-lora-tune-locks'):path=self.case/'fake-locks'/path.name
        with REAL_EXCLUSIVE(path,blocking=blocking):yield
    def execute(self,job,mode='success',**kwargs):
        def popen(command,**kw):
            self.assertEqual(command[1],str(REPO/'scripts/experiments/lib/infer_worker.py')); self.assertTrue(kw['start_new_session']); self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'1'); kw['env']['SMOKE_INFER_MODE']=mode
            return REAL_POPEN([sys.executable,'-B',str(Path(__file__).resolve()),'--fake-worker',command[-1]],**kw)
        identity={'physical_ids':[1],'uuids':['CPU-FAKE-1'],'pci_bus_ids':['00000000:03:00.0']}
        with patch.object(ex,'_gpu_identity',return_value=identity),patch.object(ex,'_exclusive',side_effect=self.locks),patch.object(ex.subprocess,'Popen',side_effect=popen):return ex.execute_infer_job(job,**kwargs)
    def test_import_cpu_only(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','peft','wandb','deepspeed','vllm'})
    def test_compile_new_files(self):
        for f in ('run_infer_job.py','lib/infer_executor.py','lib/infer_worker.py'):compile((REPO/'scripts/experiments'/f).read_text(),f,'exec')
    def test_base_six_tasks_one_engine(self):
        _,runs,jobs=self.records(); result=self.execute(jobs['infer'][0]); self.assertEqual(result['status'],'succeeded'); output=Path(result['output_dir']); summary=read_json(output/'inference_result.json'); events=read_json(output/'fake-events.json')
        self.assertEqual(events.count('load-engine'),1); self.assertEqual(events.count('shutdown'),1); self.assertEqual([x for x in events if x.startswith('infer:')],['infer:cosine_'+t+'_test' for t in TASKS]); self.assertEqual(summary['model_loads'],1); self.assertEqual(summary['task_order'],list(TASKS)); job,_=load_job(jobs['infer'][0]); self.assertEqual(len(job['artifacts']),7); self.assertEqual(load_stage_run(runs['infer'])[0]['status'],'running')
        for item in summary['tasks']:self.assertEqual(item['sample_count'],2)
        for a in job['artifacts']:check_manifest(read_json(jobs['infer'][0]/a['path']),verify_contents=True)
        for event in events:
            if event.startswith('model-view:'):self.assertFalse(Path(event.split(':',1)[1]).exists())
    def test_adapter_with_six_tasks(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); result=self.execute(jobs['infer'][0],input_manifests={'adapter':adapter}); self.assertEqual(read_json(Path(result['output_dir'])/'inference.request.json')['adapter_fingerprint'],adapter['fingerprint'])
    def test_missing_adapter_rejected(self):
        _,_,jobs=self.records('multitask')
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0])
    def test_foreign_experiment_adapter_rejected(self):
        _,_,first=self.records('multitask'); adapter=producer_fixture(first['tune'][0]); _,_,second=self.records('multitask')
        with self.assertRaises(ValueError):self.execute(second['infer'][0],input_manifests={'adapter':adapter})
    def test_unregistered_adapter_rejected(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); adapter=make_manifest('adapter',adapter['path'],producer=adapter['producer'],metadata={**adapter['metadata'],'tamper':True})
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0],input_manifests={'adapter':adapter})
    def test_tampered_adapter_payload(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); (Path(adapter['path'])/'adapter_model.safetensors').write_bytes(b'ALTERED')
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0],input_manifests={'adapter':adapter})
    def test_external_adapter_rejected(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); adapter=make_manifest('adapter',adapter['path'],source='external')
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0],input_manifests={'adapter':adapter})
    def test_unsuccessful_producer_rejected(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); record=read_json(jobs['tune'][0]/'job.json'); record['status']='failed'; write_json_atomic(jobs['tune'][0]/'job.json',record)
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0],input_manifests={'adapter':adapter})
    def test_bound_inputs_not_overridden(self):
        _,_,jobs=self.records('multitask'); adapter=producer_fixture(jobs['tune'][0]); job,snap,exp,_=ex._context(jobs['infer'][0]); ex._inputs(jobs['infer'][0],job,snap,{'adapter':adapter},exp)
        with self.assertRaisesRegex(ValueError,'overridden'):self.execute(jobs['infer'][0],input_manifests={'adapter':adapter})
    def test_frozen_yaml_not_reread(self):
        _,_,jobs=self.records(); self.yaml.write_text('invalid: setting'); self.assertEqual(self.execute(jobs['infer'][0])['status'],'succeeded')
    def test_terminal_job_not_restarted(self):
        _,_,jobs=self.records(); self.execute(jobs['infer'][0])
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0])
    def test_existing_output_preserved(self):
        _,_,jobs=self.records(); job,_=load_job(jobs['infer'][0]); Path(job['output_dir']).mkdir(parents=True)
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0])
        self.assertFalse((jobs['infer'][0]/'execution.json').exists())
    def test_existing_receipt_preserved(self):
        _,_,jobs=self.records(); (jobs['infer'][0]/'execution.json').write_text('preserve')
        with self.assertRaises(ValueError):self.execute(jobs['infer'][0])
        self.assertEqual((jobs['infer'][0]/'execution.json').read_text(),'preserve')
    def test_bad_gpu_group_index(self):
        _,_,jobs=self.records()
        for index in (-1,1,True):
            with self.assertRaises(ValueError):self.execute(jobs['infer'][0],gpu_group_index=index)
    def test_environment_strips_ddp(self):
        with patch.dict(os.environ,{'RANK':'1','WORLD_SIZE':'2','LOCAL_RANK':'1','MASTER_PORT':'29500','TORCHELASTIC_RUN_ID':'x'}):env=ex._environment([1])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        for k in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_PORT','TORCHELASTIC_RUN_ID'):self.assertNotIn(k,env)
    def test_empty_answers_preserved(self):
        _,_,jobs=self.records(); result=self.execute(jobs['infer'][0],mode='empty'); self.assertTrue(all(x['empty_predictions']==2 for x in read_json(Path(result['output_dir'])/'inference_result.json')['tasks']))
    def failure(self,mode):
        _,_,jobs=self.records()
        with self.assertRaises(RuntimeError):self.execute(jobs['infer'][0],mode=mode)
        job,_=load_job(jobs['infer'][0]); self.assertEqual(job['status'],'failed'); self.assertEqual(job['artifacts'],[]); receipt=read_json(jobs['infer'][0]/'execution.json'); self.assertTrue(receipt['cleanup_completed']); failure=read_json(Path(receipt['output_dir'])/'inference_failure.json'); self.assertEqual(failure['status'],'failed'); self.assertFalse((Path(receipt['output_dir'])/'inference_result.json').exists())
    def test_worker_failure_cleanup(self):self.failure('failure')
    def test_partial_predictions_fail(self):self.failure('partial')
    def test_gt_mismatch_fails(self):self.failure('badgt')
    def test_shutdown_failure_fails(self):self.failure('shutdown')
    def test_request_tamper_rejected(self):
        path=self.case/'request.json'; write_json_atomic(path,{'schema_version':1,'request_fingerprint':'bad'})
        with self.assertRaisesRegex(ValueError,'fingerprint'):worker._request(path)
    def test_gpu_zero_rejected_before_query(self):
        req={'schema_version':1,'physical_gpu_ids':[0],'params':{'tensor_parallel_size':1}}; req['request_fingerprint']=fingerprint(req); path=self.case/'request.json'; write_json_atomic(path,req)
        with patch.object(worker.subprocess,'run') as query:
            with self.assertRaises(ValueError):worker._request(path)
            query.assert_not_called()
    def test_processor_sampling(self):
        params=load_config(self.yaml)['config']['infer']['params']; p=types.SimpleNamespace(image_processor=types.SimpleNamespace(size={'keep':1},min_pixels=0,max_pixels=0),video_processor=types.SimpleNamespace(size={},min_pixels=0,max_pixels=0,min_frames=0,max_frames=0,fps=0,num_frames=None,do_sample_frames=False)); info=worker._configure_processor(p,params); self.assertEqual(p.image_processor.size['longest_edge'],params['max_pixels']); self.assertEqual(p.video_processor.max_frames,params['video_max_frames']); self.assertTrue(info['do_sample_frames'])
    def test_malformed_annotation_rejected(self):
        path=self.case/'bad.json'
        for data in ([],{},[{}],[{'conversations':[{'from':'gpt','value':'a'},{'from':'gpt','value':'b'}]}]):
            write_json_atomic(path,data)
            with self.assertRaises(ValueError):worker.load_samples(path,self.case)
    def test_jsonl_samples(self):
        path=self.case/'good.jsonl'; path.write_text(json.dumps({'conversations':[{'from':'gpt','value':'a'}]})+chr(10)); self.assertEqual(worker.load_samples(path,self.case)[0]['data_path'],str(self.case))
    def test_cli_relative_job_rejected(self):
        result=subprocess.run([sys.executable,'-B',str(REPO/'scripts/experiments/run_infer_job.py'),'--job-dir','relative'],capture_output=True,text=True); self.assertEqual(result.returncode,1); self.assertIn('absolute',result.stderr)
    def test_pci_order_mismatch(self):
        fake=types.SimpleNamespace(stdout='0, GPU-A, 00000000:04:00.0'+chr(10)+'1, GPU-B, 00000000:03:00.0'+chr(10))
        with patch.object(ex.subprocess,'run',return_value=fake):
            with self.assertRaisesRegex(ValueError,'PCI'):ex._gpu_identity([1],{})

def main():
    global RESULT
    parser=argparse.ArgumentParser(); parser.add_argument('--result-root',required=True,type=Path); args=parser.parse_args(); RESULT=args.result_root.resolve(); RESULT.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(InferTests); result=unittest.TextTestRunner(verbosity=2).run(suite); write_json_atomic(RESULT/'summary.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'details':[{'test':str(t),'traceback':s} for t,s in result.failures+result.errors]}); return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
