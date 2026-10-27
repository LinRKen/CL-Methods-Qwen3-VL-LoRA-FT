"""Real 4B/ZeRO-3 executor smoke on physical GPU 1, all outputs below smoke/results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
LIB=REPO/'scripts/experiments'
sys.path.insert(0,str(LIB))
import yaml
import pipeline
from lib.config import load_config,resolve_run_config
from lib.records import create_experiment,create_stage_run,create_job,load_job
from lib.common import read_json,write_json_atomic
from lib.artifacts import check_manifest
BASE=Path('/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct')
DATA=Path('/home/linruikang/data/datasets/COSINE')
PYTHON='/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python'

def setup(root):
    data=root/'dataset'; processor=root/'processor'
    processor.mkdir()
    for item in BASE.iterdir():
        if item.is_file() and not item.name.startswith('.') and not any(
                text in item.name for text in ('safetensors','pytorch_model','optimizer','training_args')):
            (processor/item.name).symlink_to(item)
    selected=[]
    for task in ('motion','hypo'):
        path=DATA/task/'train.json'; records=json.loads(path.read_text())
        indices=[]
        # Two ordinary images, one multi-image record and one video where available.
        predicates=[lambda x:len(x.get('image',[]))==1 and not x.get('video'),
                    lambda x:len(x.get('image',[]))==1 and not x.get('video'),
                    lambda x:len(x.get('image',[]))>1,
                    lambda x:bool(x.get('video'))]
        for predicate in predicates:
            match=next((i for i,x in enumerate(records) if i not in indices and predicate(x)),None)
            if match is None: match=next(i for i in range(len(records)) if i not in indices)
            indices.append(match)
        (data/task).mkdir(parents=True)
        (data/task/'train.json').write_text(json.dumps([records[i] for i in indices],ensure_ascii=False,indent=2))
        selected.append({'task':task,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'indices':indices,'ids':[records[i].get('_meta',{}).get('id') for i in indices],
                         'image_counts':[len(records[i].get('image',[])) for i in indices],
                         'video_counts':[len(records[i].get('video',[])) for i in indices]})
    write_json_atomic(root/'samples.json',selected)
    cfg={'experiment':{'name':'tune-executor-smoke','setting':'cl','dataset_root':str(data),
                      'base_model':str(BASE),'processor_path':str(processor),
                      **{k:str(root/k) for k in ('records_root','checkpoints_root','infer_root','eval_root')}},
         'tune':{'params':{'num_train_epochs':1,'per_device_train_batch_size':1,
                    'gradient_accumulation_steps':2,'lora_r':8,'lora_alpha':16,
                    'dataloader_num_workers':0,'save_strategy':'steps','save_steps':1,
                    'save_total_limit':3,'report_to':'wandb'},
                 'run':{'gpu_groups':[[1]],'master_port':29871,'wandb_mode':'offline',
                        'wandb_project':'Qwen3VL-LoRA-FT-Smoke'}}}
    setting=root/'setting.yaml'; setting.write_text(yaml.safe_dump(cfg,sort_keys=False))
    snapshot=load_config(setting)
    write_json_atomic(root/'config.snapshot.json',snapshot)
    return snapshot

def prepare(snapshot,identifier):
    exp=create_experiment(snapshot,identifier)
    plan=pipeline.build_stage_plan(resolve_run_config(snapshot,'tune'),'tune')
    run=create_stage_run(exp,'tune',run_id='smoke',stage_plan=plan)
    return [create_job(run,**item) for item in plan['jobs'][:2]]

def run(root,name,job,extra=()):
    command=[PYTHON,'-B',str(LIB/'run_tune_job.py'),'--job-dir',str(job),*extra]
    write_json_atomic(root/(name+'-command.json'),{'argv':command,'job_dir':str(job)})
    started=time.monotonic()
    with (root/(name+'.log')).open('xb') as out:
        process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                 stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
        write_json_atomic(root/(name+'-pid.json'),{'pid':process.pid})
        code=process.wait()
    write_json_atomic(root/(name+'-exit.json'),{'returncode':code,'elapsed_seconds':time.monotonic()-started})
    print(name+': exit '+str(code),flush=True)
    if code: return False
    verify_case(root,name,job)
    return True

def verify_case(root,name,job):
    record,_=load_job(job)
    manifests={item['role']:check_manifest(read_json(job/item['path']),verify_contents=True)
               for item in record['artifacts']}
    expected={'adapter','checkpoint-2','training-summary'}
    if name!='restore': expected.add('checkpoint-1')
    assert record['status']=='succeeded' and expected==set(manifests), (name,record['status'],set(manifests))
    summary=read_json(manifests['training-summary']['path'])
    assert summary['global_step']==2 and summary['train_samples']==4 and summary['world_size']==1
    assert summary['trainable_parameters']==5898240
    assert summary['epoch']==1.0
    receipt=read_json(Path(record['output_dir'])/'execution.json')
    assert receipt['status']=='succeeded' and receipt['returncode']==0
    assert receipt['gpu_identity']=={'physical_ids':[1],
                                   'uuids':['GPU-63a25b73-5c01-919f-188a-29c116904574']}
    assert receipt['environment']['CUDA_VISIBLE_DEVICES']==receipt['gpu_identity']['uuids'][0]
    for key in ('WANDB_X_STATS_GPU_DEVICE_IDS','WANDB__STATS_GPU_DEVICE_IDS'):
        assert receipt['environment'][key]=='[1]'
    command=receipt['command']
    assert ('--adapter_name_or_path' in command)==(name=='continue')
    assert ('--resume_checkpoint_path' in command)==(name=='restore')
    policies={'fresh':'base_model_with_fresh_training_state',
              'continue':'adapter_weights_with_fresh_training_state',
              'restore':'restore_full_checkpoint'}
    assert summary['training_state_policy']==policies[name]
    state=read_json(Path(manifests['checkpoint-2']['path'])/'trainer_state.json')
    assert state['global_step']==2 and state['epoch']==1.0
    assert [row['step'] for row in state['log_history'] if 'loss' in row]==[1,2]
    import math
    for row in state['log_history']:
        if 'loss' in row:
            assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0
    if name!='restore':
        first=read_json(Path(manifests['checkpoint-1']['path'])/'trainer_state.json')
        assert first['global_step']==1 and first['epoch']==0.5
        assert first['log_history'][0]['learning_rate']==0.0
    text=Path(receipt['log_path']).read_text()
    assert 'Traceback (most recent call last)' not in text and 'CUDA out of memory' not in text
    return {'successful':True,'status':record['status'],'returncode':receipt['returncode'],
            'policy':summary['training_state_policy'],'summary':summary,
            'artifact_roles':sorted(manifests),'elapsed_seconds':receipt['elapsed_seconds'],
            'pid':receipt['pid'],'launcher_pid':receipt['launcher_pid'],
            'gpu_identity':receipt['gpu_identity']}

def verify_completed(root):
    # This mode only reads existing results. It never restarts a terminal job.
    paths={name:Path(path) for name,path in read_json(root/'job-paths.json').items()}
    outcomes={name:verify_case(root,name,path) for name,path in paths.items()}
    from safetensors.torch import load_file
    import torch
    assert not torch.cuda.is_initialized()
    def tensor_file(name,role):
        manifest=read_json(paths[name]/('artifacts/'+role+'.json'))
        return Path(manifest['path'])/'adapter_model.safetensors'
    def tensors(name,role):
        return load_file(str(tensor_file(name,role)),device='cpu')
    def equal(left,right):
        return set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left)
    fresh=tensors('fresh','adapter'); continued=tensors('continue','adapter')
    first_continued=tensors('continue','checkpoint-1'); restored=tensors('restore','adapter')
    checks={
        'restored_final_tensors_equal_uninterrupted':equal(fresh,restored),
        'continued_step1_tensors_equal_parent_with_zero_first_lr':equal(fresh,first_continued),
        'continued_final_tensors_differ_from_parent':not equal(fresh,continued),
        'all_tensors_cpu_finite_language_lora':all(
            v.device.type=='cpu' and bool(torch.isfinite(v).all()) and 'visual' not in k.split('.') and 'lora_' in k
            for values in (fresh,continued,restored) for k,v in values.items()),
        'all_lora_b_tensors_nonzero':all(bool(torch.count_nonzero(v))
            for values in (fresh,continued,restored) for k,v in values.items() if 'lora_B' in k),
        'cuda_not_initialized_during_audit':not torch.cuda.is_initialized(),
    }
    fresh_state=read_json(tensor_file('fresh','checkpoint-2').parent/'trainer_state.json')
    restored_state=read_json(tensor_file('restore','checkpoint-2').parent/'trainer_state.json')
    checks['restored_training_history_equal_uninterrupted']=fresh_state['log_history']==restored_state['log_history']
    for row in read_json(root/'samples.json'):
        checks['source_annotation_unchanged_'+row['task']]=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()==row['source_sha256']
    receipts=[read_json(Path(load_job(path)[0]['output_dir'])/'execution.json') for path in paths.values()]
    checks['unique_wandb_runs']=len({x['environment']['WANDB_RUN_ID'] for x in receipts})==3
    for receipt in receipts:
        for key in ('pid','launcher_pid'):
            try: os.kill(receipt[key],0)
            except ProcessLookupError: pass
            else: checks['all_recorded_workers_exited']=False
    checks.setdefault('all_recorded_workers_exited',True)
    successful=all(item['successful'] for item in outcomes.values()) and all(checks.values())
    evidence={'successful':successful,'cases':outcomes,'checks':checks,
              'adapter_tensor_count':len(fresh),
              'lora_b_tensor_count':sum('lora_B' in k for k in fresh),
              'audit_only':True}
    target=root/'outcomes.verified.json'
    if target.exists(): raise ValueError('Refusing to overwrite audit evidence')
    write_json_atomic(target,evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2),flush=True)
    return 0 if successful else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--result-dir',required=True,type=Path)
    parser.add_argument('--verify-only',action='store_true',help='Audit existing results without running training')
    args=parser.parse_args(); root=args.result_dir.resolve(strict=True)
    if not root.is_relative_to((REPO/'smoke/results').resolve(strict=True)): parser.error('smoke/results required')
    if args.verify_only:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        return verify_completed(root)
    if (root/'config.snapshot.json').exists(): parser.error('Refusing evidence reuse')
    snapshot=setup(root)
    jobs=prepare(snapshot,'smoke-source')
    recovery=prepare(snapshot,'smoke-recovery')
    write_json_atomic(root/'job-paths.json',{'fresh':str(jobs[0]),'continue':str(jobs[1]),'restore':str(recovery[0])})
    outcomes={}
    outcomes['fresh']=run(root,'fresh',jobs[0])
    if outcomes['fresh']:
        outcomes['continue']=run(root,'continue',jobs[1],['--input','parent-motion='+str(jobs[0]/'artifacts/adapter.json')])
        outcomes['restore']=run(root,'restore',recovery[0],['--resume-training-state','--input',
                                               'recovery='+str(jobs[0]/'artifacts/checkpoint-1.json')])
    write_json_atomic(root/'outcomes.json',outcomes)
    return 0 if len(outcomes)==3 and all(outcomes.values()) else 1
if __name__=='__main__': raise SystemExit(main())
