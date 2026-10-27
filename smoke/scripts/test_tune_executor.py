"""CPU acceptance for the real tune executor; simulated payloads are never models."""
import argparse
from copy import deepcopy
import importlib.abc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LIB = REPO / 'scripts/experiments'

def fake_worker():
    # Deliberately fake bytes exercise registration only, never model correctness.
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    output = Path(args['--output_dir'])
    adapter = Path(args['--final_adapter_dir'])
    checkpoint = output / 'checkpoint-1'
    config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
              'r': int(args['--lora_r']), 'lora_alpha': int(args['--lora_alpha']),
              'lora_dropout': float(args['--lora_dropout']),
              'target_modules': ['q_proj','k_proj','v_proj','o_proj'],
              'exclude_modules': r'.*visual[.].*', 'bias':'none',
              'base_model_name_or_path':args['--model_name_or_path']}
    for path in (adapter, checkpoint):
        path.mkdir(parents=True)
        (path/'adapter_config.json').write_text(json.dumps(config))
        (path/'adapter_model.safetensors').write_bytes(b'FAKE CPU REGISTRATION ONLY')
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth'):
        (checkpoint/name).write_bytes(b'FAKE STATE')
    (checkpoint/'trainer_state.json').write_text('{"global_step":1}')
    if '--deepspeed' in args:
        (checkpoint/'latest').write_text('global_step1')
        state=checkpoint/'global_step1'; state.mkdir()
        for name in ('zero_optim_states.pt','mp_model_states.pt'):
            (state/name).write_bytes(b'FAKE DS STATE')
    policy = 'restore_full_checkpoint' if '--resume_checkpoint_path' in args else (
        'adapter_weights_with_fresh_training_state' if '--adapter_name_or_path' in args
        else 'base_model_with_fresh_training_state')
    summary = {'global_step':1,'world_size':1,'train_samples':2,'trainable_parameters':128,
               'training_state_policy':policy,'final_adapter_dir':str(adapter)}
    (output/'training_result.json').write_text(json.dumps(summary))
    (output/'worker.json').write_text(json.dumps({'pid':os.getpid(),'args':args,
               'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),'policy':policy}))
    if os.environ.get('SMOKE_FAKE_MODE') == 'incomplete':
        (checkpoint/'scheduler.pt').unlink()
    if os.environ.get('SMOKE_FAKE_MODE') == 'failure':
        return 17
    return 0

if len(sys.argv)>1 and sys.argv[1]=='--fake-worker':
    raise SystemExit(fake_worker())

os.environ['CUDA_VISIBLE_DEVICES']=''
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','peft','wandb','deepspeed','vllm'}:
            raise ImportError('CPU test forbids model import: '+fullname)
sys.meta_path.insert(0,BlockModels())
sys.path.insert(0,str(LIB))
import yaml
import pipeline
from lib import tune_executor as ex
from lib.common import read_json, write_json_atomic
from lib.config import load_config, resolve_run_config, deep_merge
from lib.artifacts import make_manifest
from lib.records import (create_experiment,create_stage_run,create_job,load_job,
                         load_stage_run,update_status,register_artifact)
RESULT=None
REAL_POPEN=subprocess.Popen

class TuneTests(unittest.TestCase):
    def setUp(self):
        self.case=RESULT/'cases'/self._testMethodName
        self.case.mkdir(parents=True)
        self.sequence=0
        base=self.case/'base'
        base.mkdir(); (base/'config.json').write_text('{}')
        (base/'preprocessor_config.json').write_text('{}')
        data=self.case/'data'
        for task in ('motion','hypo','pattern','nav','view','geo','whole'):
            (data/task).mkdir(parents=True); (data/task/'train.json').write_text('[]')
        ds=self.case/'ds.json'; ds.write_text('{}')
        doc={'experiment':{'name':'smoke','setting':'cl','base_model':str(base),
             'processor_path':str(base),'dataset_root':str(data),
             **{k:str(self.case/k) for k in ('records_root','checkpoints_root','infer_root','eval_root')}},
             'tune':{'params':{'deepspeed':str(ds),'lora_r':8,'lora_alpha':16},
                     'run':{'gpu_groups':[[1]],'wandb_mode':'offline'}}}
        self.yaml=self.case/'setting.yaml'; self.yaml.write_text(yaml.safe_dump(doc))
        self.snapshot=load_config(self.yaml)
        self.jobs=[]
    def records(self,snapshot=None,ids=None):
        self.sequence+=1
        snapshot=snapshot or self.snapshot
        exp=create_experiment(snapshot,'exp-'+str(self.sequence))
        effective=resolve_run_config(snapshot,'tune')
        plan=pipeline.build_stage_plan(effective,'tune')
        run=create_stage_run(exp,'tune',run_id='run',stage_plan=plan)
        jobs=[]
        for item in plan['jobs'][:ids or 2]:
            jobs.append(create_job(run,**item))
        self.jobs=jobs
        return exp,run,jobs
    def identity(self,config,env,group):
        ids=['GPU-CPU-FAKE-'+str(i) for i in group]
        env['CUDA_VISIBLE_DEVICES']=','.join(ids)
        for k in ('WANDB_X_STATS_GPU_DEVICE_IDS','WANDB__STATS_GPU_DEVICE_IDS'):
            env[k]=json.dumps(group)
        return {'physical_ids':group,'uuids':ids}
    def popen(self,command,**kwargs):
        self.assertEqual(command[1:3],['-m','torch.distributed.run'])
        self.assertTrue(kwargs['start_new_session'])
        start=next(i for i,a in enumerate(command) if a.endswith('/train_qwen.py'))+1
        return REAL_POPEN([sys.executable,'-B',str(Path(__file__).resolve()),'--fake-worker',*command[start:]],**kwargs)
    def execute(self,job,**kwargs):
        with patch.object(ex,'_gpu_identity',side_effect=self.identity), patch.object(ex.subprocess,'Popen',side_effect=self.popen):
            return ex.execute_tune_job(job,**kwargs)
    def adapter(self,job):
        return read_json(job/'artifacts/adapter.json')
    def checkpoint(self,job):
        return read_json(job/'artifacts/checkpoint-1.json')
    def test_import_is_cpu_only(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','peft','wandb','deepspeed','vllm'})
    def test_syntax_all_changed_sources(self):
        for name in ('scripts/experiments/lib/tune_executor.py','scripts/experiments/run_tune_job.py',
                     'qwen-vl-finetune/qwenvl/train/argument.py','qwen-vl-finetune/qwenvl/train/train_qwen.py',
                     'qwen-vl-finetune/qwenvl/data/data_processor.py'):
            compile((REPO/name).read_text(),name,'exec')
    def test_fresh_success_and_registered_outputs(self):
        _,run,jobs=self.records()
        receipt=self.execute(jobs[0])
        self.assertEqual(receipt['status'],'succeeded')
        job,_=load_job(jobs[0]); self.assertEqual(job['status'],'succeeded')
        self.assertEqual({a['role'] for a in job['artifacts']},{'adapter','checkpoint-1','training-summary'})
        self.assertEqual(load_stage_run(run)[0]['status'],'running')
        self.assertEqual(receipt['environment']['CUDA_VISIBLE_DEVICES'],'GPU-CPU-FAKE-1')
        self.assertEqual(receipt['environment']['WANDB_X_STATS_GPU_DEVICE_IDS'],'[1]')
    def test_continue_uses_adapter_and_fresh_state(self):
        _,_,jobs=self.records(); self.execute(jobs[0])
        result=self.execute(jobs[1],input_manifests={'parent-motion':self.adapter(jobs[0])})
        command=result['command']; self.assertIn('--adapter_name_or_path',command)
        self.assertNotIn('--resume_checkpoint_path',command)
        self.assertEqual(result['training_state_policy'],ex.CONTINUE)
    def test_dependency_not_succeeded(self):
        _,_,jobs=self.records()
        with self.assertRaisesRegex(ValueError,'dependency'):
            self.execute(jobs[1])
        self.assertFalse((jobs[1]/'execution.json').exists())
    def test_missing_parent_rejected_before_launch(self):
        _,_,jobs=self.records(); self.execute(jobs[0])
        with self.assertRaisesRegex(ValueError,'required input'):
            self.execute(jobs[1])
        self.assertEqual(load_job(jobs[1])[0]['status'],'failed')
    def test_parent_content_tamper(self):
        _,_,jobs=self.records(); self.execute(jobs[0]); manifest=self.adapter(jobs[0])
        (Path(manifest['path'])/'adapter_model.safetensors').write_bytes(b'TAMPERED')
        with self.assertRaisesRegex(ValueError,'contents'):
            self.execute(jobs[1],input_manifests={'parent':manifest})
    def test_unregistered_parent_rejected(self):
        _,_,jobs=self.records(); self.execute(jobs[0]); m=self.adapter(jobs[0])
        m=make_manifest('adapter',m['path'],producer=m['producer'],metadata={**m['metadata'],'extra':1})
        with self.assertRaisesRegex(ValueError,'not registered'):
            self.execute(jobs[1],input_manifests={'parent':m})
    def test_external_adapter_not_accepted(self):
        _,_,jobs=self.records(); self.execute(jobs[0]); m=self.adapter(jobs[0])
        m=make_manifest('adapter',m['path'],source='external',metadata=m['metadata'])
        with self.assertRaises(ValueError): self.execute(jobs[1],input_manifests={'parent':m})
    def test_full_restore_same_configuration(self):
        _,_,first=self.records(); self.execute(first[0]); checkpoint=self.checkpoint(first[0])
        _,_,second=self.records(); result=self.execute(second[0],input_manifests={'recovery':checkpoint},resume_training_state=True)
        self.assertEqual(result['training_state_policy'],ex.RESTORE)
        self.assertIn('--resume_checkpoint_path',result['command'])
        self.assertNotIn('--adapter_name_or_path',result['command'])
    def test_checkpoint_without_restore_flag(self):
        _,_,first=self.records(); self.execute(first[0]); m=self.checkpoint(first[0])
        _,_,second=self.records()
        with self.assertRaisesRegex(ValueError,'Full restore'):
            self.execute(second[0],input_manifests={'recovery':m})
    def test_restore_without_checkpoint(self):
        _,_,jobs=self.records()
        with self.assertRaisesRegex(ValueError,'Full restore'):
            self.execute(jobs[0],resume_training_state=True)
    def test_restore_world_size_changed(self):
        _,_,first=self.records(); self.execute(first[0]); m=self.checkpoint(first[0])
        _,_,second=self.records(); job,snapshot,exp,_=ex._context(second[0])
        with self.assertRaisesRegex(ValueError,'world size'):
            ex._inputs(job,snapshot,{'recovery':m},True,exp,2)
    def test_restore_wrong_task(self):
        _,_,first=self.records(); self.execute(first[0]); m=self.checkpoint(first[0])
        self.execute(first[1],input_manifests={'parent':self.adapter(first[0])})
        _,_,second=self.records()
        self.execute(second[0])
        with self.assertRaisesRegex(ValueError,'same task'):
            self.execute(second[1],input_manifests={'parent':self.adapter(second[0]),'recovery':m},resume_training_state=True)
    def test_repeat_terminal_job_refused(self):
        _,_,jobs=self.records(); self.execute(jobs[0])
        with self.assertRaisesRegex(ValueError,'planned'):
            self.execute(jobs[0])
    def test_output_reuse_refused(self):
        _,_,jobs=self.records(); output=Path(load_job(jobs[0])[0]['output_dir'])
        output.mkdir(parents=True); (output/'sentinel').write_text('unchanged')
        with self.assertRaisesRegex(ValueError,'existing'):
            self.execute(jobs[0])
        self.assertEqual((output/'sentinel').read_text(),'unchanged')
        self.assertFalse((jobs[0]/'execution.json').exists())
    def test_gpu_group_bad_index(self):
        _,_,jobs=self.records()
        for index in (-1,1,True,'0'):
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'gpu_group_index'):
                self.execute(jobs[0],gpu_group_index=index)
    def test_failure_preserves_complete_checkpoint(self):
        _,_,jobs=self.records()
        with patch.dict(os.environ,{'SMOKE_FAKE_MODE':'failure'}),self.assertRaisesRegex(RuntimeError,'17'):
            self.execute(jobs[0])
        job,_=load_job(jobs[0]); self.assertEqual(job['status'],'failed')
        self.assertEqual([i['role'] for i in job['artifacts']],['checkpoint-1'])
        receipt=read_json(jobs[0]/'execution.json')
        self.assertEqual(receipt['returncode'],17)
        with self.assertRaises(ProcessLookupError): os.kill(receipt['launcher_pid'],0)
    def test_failed_checkpoint_can_restore(self):
        _,_,first=self.records()
        with patch.dict(os.environ,{'SMOKE_FAKE_MODE':'failure'}),self.assertRaises(RuntimeError): self.execute(first[0])
        _,_,second=self.records()
        result=self.execute(second[0],input_manifests={'recovery':self.checkpoint(first[0])},resume_training_state=True)
        self.assertEqual(result['status'],'succeeded')
    def test_partial_checkpoint_prevents_success(self):
        _,_,jobs=self.records()
        with patch.dict(os.environ,{'SMOKE_FAKE_MODE':'incomplete'}),self.assertRaisesRegex(ValueError,'incomplete'):
            self.execute(jobs[0])
        job,_=load_job(jobs[0]); self.assertEqual(job['status'],'failed'); self.assertEqual(job['artifacts'],[])
    def test_adapter_compatibility_checks(self):
        _,_,jobs=self.records(); self.execute(jobs[0]); m=self.adapter(jobs[0]); path=Path(m['path'])
        original=read_json(path/'adapter_config.json'); params=self.snapshot['config']['tune']['params']
        for change in ({'r':9},{'lora_alpha':99},{'lora_dropout':0.1},{'target_modules':['q_proj']},
                       {'exclude_modules':None},{'bias':'all'},{'modules_to_save':['lm_head']},
                       {'base_model_name_or_path':str(self.case/'other')},{'use_dora':True}):
            with self.subTest(change=change):
                write_json_atomic(path/'adapter_config.json',{**original,**change})
                with self.assertRaises(ValueError): ex._adapter_config(path,params,self.snapshot['config']['experiment']['base_model'])
        write_json_atomic(path/'adapter_config.json',original)
    def test_native_checkpoint_layout(self):
        p=self.case/'checkpoint'; p.mkdir(); (p/'trainer_state.json').write_text('{"global_step":2}')
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth'): (p/name).write_bytes(b'STATE')
        self.assertEqual(ex._checkpoint_state(p,1,'',False)['global_step'],2)
        with self.assertRaises(ValueError): ex._checkpoint_state(p,1,'',True)
        (p/'scaler.pt').write_bytes(b'SCALER'); ex._checkpoint_state(p,1,'',True)
        with self.assertRaises(ValueError): ex._checkpoint_state(p,2,'',False)
    def test_deepspeed_checkpoint_scheduler_requirement(self):
        p=self.case/'checkpoint'; p.mkdir(); ds=self.case/'ds.json'
        (p/'trainer_state.json').write_text('{"global_step":2}')
        (p/'latest').write_text('global_step2'); (p/'rng_state.pth').write_bytes(b'RNG')
        state=p/'global_step2'; state.mkdir()
        for name in ('zero_optim_states.pt','mp_model_states.pt'): (state/name).write_bytes(b'DS')
        ds.write_text('{}')
        with self.assertRaisesRegex(ValueError,'incomplete'): ex._checkpoint_state(p,1,ds,False)
        (p/'scheduler.pt').write_bytes(b'HF SCHEDULER'); ex._checkpoint_state(p,1,ds,False)
        (p/'scheduler.pt').unlink(); ds.write_text('{"optimizer":{"type":"AdamW"}}')
        ex._checkpoint_state(p,1,ds,False)
        (p/'latest').write_text('global_step999')
        with self.assertRaisesRegex(ValueError,'tag'): ex._checkpoint_state(p,1,ds,False)
    def test_environment_clears_distributed_state(self):
        job={'experiment_id':'e','run_id':'r','job_id':'j'}
        with patch.dict(os.environ,{'RANK':'99','WORLD_SIZE':'99','MASTER_PORT':'99','TORCHELASTIC_RUN_ID':'old'}):
            env=ex._environment(self.snapshot['config'],job,self.case,[1])
        self.assertFalse(set(env)&{'RANK','WORLD_SIZE','MASTER_PORT','TORCHELASTIC_RUN_ID'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        self.assertEqual(env['WANDB_RESUME'],'never')
        other=ex._environment(self.snapshot['config'],job,self.case,[1])
        self.assertNotEqual(env['WANDB_RUN_ID'],other['WANDB_RUN_ID'])
    def test_job_lock_rejects_duplicate(self):
        _,_,jobs=self.records()
        with ex._exclusive(jobs[0]/'.execution.lock'):
            with self.assertRaisesRegex(ValueError,'Another executor'): self.execute(jobs[0])
    def test_stage_slot_released_on_error(self):
        directory=self.case/'run'
        with self.assertRaises(RuntimeError):
            with ex._stage_slot(directory,1): raise RuntimeError('test')
        with ex._stage_slot(directory,1) as slot: self.assertEqual(slot,0)
    def test_cli_rejects_relative_job(self):
        p=subprocess.run([sys.executable,'-B',str(LIB/'run_tune_job.py'),'--job-dir','relative'],capture_output=True,text=True)
        self.assertEqual(p.returncode,1); self.assertIn('absolute',p.stderr)
    def test_cli_rejects_duplicate_inputs(self):
        f=self.case/'manifest.json'; f.write_text('{}')
        p=subprocess.run([sys.executable,'-B',str(LIB/'run_tune_job.py'),'--job-dir',str(self.case),
                          '--input','a='+str(f),'--input','a='+str(f)],capture_output=True,text=True)
        self.assertEqual(p.returncode,1); self.assertIn('unique',p.stderr)

class Results(unittest.TextTestResult):
    def startTest(self,test):
        self.started=time.monotonic(); super().startTest(test)
    def addSuccess(self,test):
        super().addSuccess(test); self.rows.append({'test':test.id(),'status':'passed','seconds':time.monotonic()-self.started})
    def addFailure(self,test,err):
        super().addFailure(test,err); self.rows.append({'test':test.id(),'status':'failed','traceback':self._exc_info_to_string(err,test)})
    def addError(self,test,err):
        super().addError(test,err); self.rows.append({'test':test.id(),'status':'error','traceback':self._exc_info_to_string(err,test)})
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.rows=[]

def main():
    global RESULT
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--result-dir',required=True,type=Path)
    args=parser.parse_args(); RESULT=args.result_dir.resolve(strict=True)
    if not RESULT.is_relative_to((REPO/'smoke/results').resolve(strict=True)): parser.error('Results must stay in smoke/results')
    if (RESULT/'cases').exists() or (RESULT/'tests.json').exists(): parser.error('Refusing existing test evidence')
    runner=unittest.TextTestRunner(verbosity=2,resultclass=Results)
    result=runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TuneTests))
    (RESULT/'tests.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'successful':result.wasSuccessful(),'cases':result.rows},ensure_ascii=False,indent=2))
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__': raise SystemExit(main())
