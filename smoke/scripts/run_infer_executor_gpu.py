'''Run base and adopted real LoRA inference on physical GPU 1, no training.'''
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/experiments'))
import yaml
import pipeline
from lib.artifacts import make_manifest,check_manifest
from lib.common import TASKS,read_json,write_json_atomic,sha256_file
from lib.config import load_config,resolve_run_config
from lib.records import create_experiment,create_stage_run,create_job,load_job,load_stage_run,bind_inputs,update_status,register_artifact
from lib.infer_worker import inspect_predictions,load_samples
BASE=Path('/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct')
DATA=Path('/home/linruikang/data/datasets/COSINE')
PYTHON='/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025/bin/python'
UUID='GPU-63a25b73-5c01-919f-188a-29c116904574'

def gpu_query():
    def query(kind,fields):
        return list(csv.reader(subprocess.check_output(['nvidia-smi','-i','1','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines(),skipinitialspace=True))
    cards=query('gpu','index,uuid,memory.used,memory.total'); processes=query('compute-apps','pid,gpu_uuid,used_memory')
    if len(cards)!=1 or cards[0][:2]!=['1',UUID] or any(row[1]!=UUID for row in processes):raise ValueError('GPU identity mismatch')
    return {'time':time.time(),'gpu':cards[0],'processes':processes}

def register_adopted_adapter(job_dir,source_manifest):
    # Explicit fixture provenance: source is a previous real smoke training result.
    source=read_json(source_manifest); check_manifest(source,verify_contents=True)
    source_job_dir=source_manifest.parent.parent; source_job,source_snap=load_job(source_job_dir)
    if source_job['status']!='succeeded' or source_job['task']!='whole':raise ValueError('Real adapter producer not successful whole tune')
    job,snap=load_job(job_dir); source_cfg=read_json(Path(source['path'])/'adapter_config.json')
    for key in ('r','lora_alpha','lora_dropout'):
        argument={'r':'lora_r','lora_alpha':'lora_alpha','lora_dropout':'lora_dropout'}[key]
        if source_cfg[key]!=snap['config']['tune']['params'][argument]:raise ValueError('Adoption parameter mismatch')
    inputs={s['role']:make_manifest(s['kind'],s['path'],source=s['source'],metadata=s['metadata']) for s in job['input_specs']}
    if source['metadata']['base_model_fingerprint']!=inputs['base-model']['fingerprint']:raise ValueError('Adoption base identity mismatch')
    bind_inputs(job_dir,inputs); update_status(job_dir,'running','Smoke fixture adopts verified existing adapter; no new training')
    adapter=Path(job['output_dir'])/'adapter'; adapter.mkdir(parents=True)
    for name in ('adapter_config.json','adapter_model.safetensors'):(adapter/name).symlink_to(Path(source['path'])/name)
    metadata={'slot':job['output_spec']['slot'],'training_task':job['task'],'lineage':job['lineage'],'training_state_policy':job['training_state_policy'],'config_fingerprint':job['config_fingerprint'],'base_model_fingerprint':inputs['base-model']['fingerprint'],'input_fingerprints':{k:v['fingerprint'] for k,v in inputs.items()},'smoke_fixture':True,'training_executed':False,'adopted_from_manifest':str(source_manifest),'adopted_from_fingerprint':source['fingerprint'],'adopted_from_producer':source['producer']}
    manifest=make_manifest('adapter',adapter,producer={k:job[k] for k in ('experiment_id','stage','run_id','job_id')},metadata=metadata); register_artifact(job_dir,'adapter',manifest); update_status(job_dir,'succeeded','Verified existing real adapter adopted for inference fixture only')
    write_json_atomic(job_dir/'adoption-evidence.json',{'source_manifest':str(source_manifest),'source_fingerprint':source['fingerprint'],'source_producer':source['producer'],'source_status':source_job['status'],'source_weight_sha256':sha256_file(Path(source['path'])/'adapter_model.safetensors'),'training_executed':False})
    return job_dir/'artifacts/adapter.json'

def setup(root):
    root.mkdir(parents=True,exist_ok=False); sample_evidence=[]
    for task in (*TASKS,'whole'):
        target=root/'dataset'/task; target.mkdir(parents=True)
        write_json_atomic(target/'train.json',[])
        if task=='whole':continue
        original=DATA/task/'test.json'; records=read_json(original); selected=[]
        predicates=[lambda s:len(s.get('image',[]))==1 and not s.get('video'),lambda s:len(s.get('image',[]))>1,lambda s:bool(s.get('video'))]
        for predicate in predicates:
            index=next((i for i,s in enumerate(records) if i not in selected and predicate(s)),None)
            if index is not None:selected.append(index)
        for index in range(len(records)):
            if len(selected)>=4:break
            if index not in selected:selected.append(index)
        selected=selected[:4]; write_json_atomic(target/'test.json',[records[i] for i in selected]); sample_evidence.append({'task':task,'source':str(original),'source_sha256':sha256_file(original),'indices':selected,'image_counts':[len(records[i].get('image',[])) for i in selected],'video_counts':[len(records[i].get('video',[])) for i in selected]})
    write_json_atomic(root/'samples.json',sample_evidence)
    jobs=[]
    for setting in ('zero_shot','multitask'):
        doc={'experiment':{'name':'infer-executor-smoke','setting':setting,'enabled_stages':['infer','eval'] if setting=='zero_shot' else ['tune','infer','eval'],'dataset_root':str(root/'dataset'),'base_model':str(BASE),'processor_path':str(BASE),**{k:str(root/k) for k in ('records_root','checkpoints_root','infer_root','eval_root')}},'infer':{'run':{'python':PYTHON,'gpu_groups':[[1]]},'params':{'batch_size':4,'max_model_len':32768,'max_new_tokens':32}},'tune':{'params':{'lora_r':8,'lora_alpha':16}}}
        yaml_path=root/(setting+'.yaml'); yaml_path.write_text(yaml.safe_dump(doc)); snap=load_config(yaml_path); exp=create_experiment(snap,'smoke-'+setting); tune_job=None
        for stage in (('infer',) if setting=='zero_shot' else ('tune','infer')):
            plan=pipeline.build_stage_plan(resolve_run_config(snap,stage),stage); run=create_stage_run(exp,stage,run_id=stage+'-smoke',stage_plan=plan); created=[create_job(run,**item) for item in plan['jobs']]
            if stage=='tune':tune_job=created[0]
            else:infer_job=created[0]
        args=[PYTHON,'-B',str(REPO/'scripts/experiments/run_infer_job.py'),'--job-dir',str(infer_job),'--gpu-group-index','0']
        if tune_job:
            source=REPO/'smoke/results/tune-orchestrator-20261027-tfhEWVk4/gpu-real/records_root/smoke-multitask/stage-runs/tune/tune-20261027T110730Z-9e4bbbb1/jobs/tune-whole/artifacts/adapter.json'
            adapter=register_adopted_adapter(tune_job,source); args+=['--input','adapter='+str(adapter)]
        jobs.append({'setting':setting,'job_dir':str(infer_job),'command':args})
    write_json_atomic(root/'cases.json',jobs); return jobs

def run(root,cases):
    initial=gpu_query(); write_json_atomic(root/'gpu-before.json',initial)
    if initial['processes']:raise RuntimeError('Physical GPU 1 already occupied')
    result=[]
    for case in cases:
        observations=[]; stop=threading.Event(); name=case['setting']; started=time.monotonic()
        def monitor():
            with (root/(name+'-gpu.jsonl')).open('x') as handle:
                while not stop.is_set():
                    try:row=gpu_query()
                    except Exception as error:row={'time':time.time(),'error':repr(error)}
                    observations.append(row); handle.write(json.dumps(row)+chr(10)); handle.flush(); stop.wait(2)
        thread=threading.Thread(target=monitor,daemon=True)
        with (root/(name+'-cli.log')).open('xb') as log:
            proc=subprocess.Popen(case['command'],cwd=REPO,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write_json_atomic(root/(name+'-launcher.json'),{'pid':proc.pid,'command':case['command']}); thread.start()
            try:code=proc.wait()
            except BaseException:proc.terminate(); proc.wait(); raise
            finally:stop.set(); thread.join(timeout=12)
        evidence={'setting':name,'returncode':code,'elapsed_seconds':time.monotonic()-started,'observations':len(observations),'observed_max_whole_gpu_mib':max((float(x['gpu'][2]) for x in observations if 'gpu' in x),default=None)}; write_json_atomic(root/(name+'-exit.json'),evidence); result.append(evidence); print(json.dumps(evidence),flush=True)
        if code:raise RuntimeError(name+' failed; preserve evidence and do not automatically repeat')
    write_json_atomic(root/'gpu-exits.json',result)

def audit(root):
    checks={}; pids=set(); receipts=[]
    for case in read_json(root/'cases.json'):
        name=case['setting']; directory=Path(case['job_dir']); job,snap=load_job(directory); receipt=read_json(directory/'execution.json'); output=Path(job['output_dir']); request=read_json(output/'inference.request.json'); summary=read_json(output/'inference_result.json'); receipts.append(receipt)
        for key,passed in {'job_succeeded':job['status']=='succeeded','receipt_succeeded':receipt['status']=='succeeded','engine_once':summary['model_loads']==1,'shutdown':summary['engine_shutdown'] is True,'physical_gpu_1':summary['physical_gpu_ids']==[1] and summary['gpu_identity']['uuids']==[UUID],'six_tasks':summary['task_order']==list(TASKS),'all_samples':len(summary['tasks'])==6 and all(t['sample_count']==4 for t in summary['tasks']),'nonempty':all(t['empty_predictions']==0 for t in summary['tasks']),'seven_artifacts':len(job['artifacts'])==7,'cleanup':receipt.get('cleanup_completed') is True,'parent_running':load_stage_run(directory.parent.parent)[0]['status']=='running','correct_adapter_mode':bool(request['adapter_path'])==(name=='multitask')}.items():checks[name+':'+key]=passed
        pids.update(receipt[k] for k in ('pid','launcher_pid') if k in receipt); pids.add(read_json(root/(name+'-launcher.json'))['pid'])
        for spec,expected in zip(request['tasks'],summary['tasks']):checks[name+':'+spec['task']+':prediction_integrity']=inspect_predictions(spec,load_samples(spec['annotation_path'],request['media_root']),request['params'])==expected
        for entry in job['artifacts']:
            manifest=read_json(directory/entry['path']); check_manifest(manifest,verify_contents=True); checks[name+':manifest:'+entry['role']]=manifest['fingerprint']==entry['fingerprint']
        for role,manifest in job['inputs'].items():check_manifest(manifest,verify_contents=True); checks[name+':input:'+role]=True
        log=(output/'logs/infer.log').read_text(errors='replace'); checks[name+':no_traceback_oom']='Traceback (most recent call last)' not in log and 'CUDA out of memory' not in log
    for evidence in read_json(root/'samples.json'):checks['original_annotation:'+evidence['task']]=sha256_file(evidence['source'])==evidence['source_sha256']
    for pid in sorted(pids):checks['pid_exited:'+str(pid)]=not Path('/proc/'+str(pid)).exists()
    final=gpu_query(); checks['gpu_1_idle']=not final['processes']; checks['gpu_1_memory_released']=float(final['gpu'][2])<50
    write_json_atomic(root/'gpu-after.json',final); summary={'successful':all(checks.values()),'checks':checks,'checks_count':len(checks),'pids':sorted(pids),'cases':read_json(root/'gpu-exits.json')}; write_json_atomic(root/'verification.json',summary); print(json.dumps({'successful':summary['successful'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v]},indent=2)); return summary['successful']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--result-root',required=True,type=Path); p.add_argument('--audit-only',action='store_true'); args=p.parse_args(); root=args.result_root.resolve()
    if not args.audit_only:run(root,setup(root))
    return 0 if audit(root) else 1
if __name__=='__main__':raise SystemExit(main())
