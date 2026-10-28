'''Exercise the production high-level frozen-run CLI on physical GPU 1 only.
Adopt an existing real 4B adapter explicitly; this smoke run performs no training.
'''
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/experiments'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline
import run_infer_executor_gpu as low
from lib.common import TASKS,read_json,write_json_atomic,sha256_file
from lib.config import resolve_run_config
from lib.records import load_experiment,load_job,load_stage_run,record_status,update_status,create_stage_run,create_job
from lib.artifacts import check_manifest
from lib import infer_orchestrator as orch

def setup(root):
    cases=low.setup(root)
    for case in cases:
        directory=Path(case['job_dir']); run_dir=directory.parent.parent; exp_dir=run_dir.parents[2]
        experiment,snapshot=load_experiment(exp_dir)
        if not any(r['stage']=='eval' for r in experiment['stage_runs']):
            plan=pipeline.build_stage_plan(resolve_run_config(snapshot,'eval'),'eval')
            eval_dir=create_stage_run(exp_dir,'eval',run_id='eval-smoke',stage_plan=plan)
            for item in plan['jobs']:create_job(eval_dir,**item)
        experiment,_=load_experiment(exp_dir)
        for entry in experiment['stage_runs']:
            if entry['stage']=='tune':
                tune=exp_dir/entry['path']; update_status(tune,'running','Adopted existing real adapter; no new training'); update_status(tune,'succeeded','Explicit adoption fixture only')
        case.update(run_dir=str(run_dir),experiment_dir=str(exp_dir),command=[low.PYTHON,'-B',str(REPO/'scripts/experiments/run_infer.py'),'--run-dir',str(run_dir)])
    write_json_atomic(root/'high-cases.json',cases)
    return cases

def audit(root):
    checks={}; pids=set(); totals=[]; warnings=[]
    def check(name,passed):checks[name]=bool(passed)
    for case in read_json(root/'high-cases.json'):
        name=case['setting']; run_dir=Path(case['run_dir']); exp_dir=Path(case['experiment_dir']); run,snapshot=load_stage_run(run_dir); summary=read_json(run_dir/'infer-summary.json'); invocation=Path(summary['invocation_dir']); execution=read_json(invocation/'execution.json'); experiment,_=load_experiment(exp_dir)
        for key,passed in {'cli_exit_zero':read_json(root/(name+'-exit.json'))['returncode']==0,'stage_succeeded':run['status']=='succeeded','summary_succeeded':summary['status']=='succeeded','receipt_succeeded':execution['status']=='succeeded','one_low_job':len(summary['jobs'])==len(execution['launches'])==1,'summary_identical':summary==read_json(invocation/'summary.json'),'experiment_active':experiment['status']=='active','eval_not_launched':all(record_status(exp_dir/r['path'])['status']=='planned' for r in experiment['stage_runs'] if r['stage']=='eval'),'frozen_config':summary['config_fingerprint']==run['config_fingerprint'],'one_invocation':len(list((run_dir/'orchestrations').iterdir()))==1}.items():check(name+':'+key,passed)
        pids.add(execution['pid']); pids.add(read_json(root/(name+'-launcher.json'))['pid'])
        predictions=0
        for entry in summary['jobs']:
            directory=run_dir/'jobs'/entry['job_id']; job,job_snap=load_job(directory); out=Path(job['output_dir']); receipt=read_json(directory/'execution.json'); request=read_json(directory/'inference.request.json'); result=read_json(out/'inference_result.json')
            info={'directory':directory,'record':job,'snapshot':job_snap,'resources':orch._resources(job_snap['config'],job['job_id']),'adapter':entry['adapter']}
            orch._audit_succeeded_job(info,snapshot['config']['experiment']['dataset_root']); check(name+':complete_parent_success_audit',True)
            for key,passed in {'job_succeeded':job['status']=='succeeded','receipt_success':receipt['status']=='succeeded','physical_1_only':entry['assignment']['physical_gpu_ids']==receipt['physical_gpu_ids']==request['physical_gpu_ids']==[1],'uuid_correct':receipt['gpu_identity']['uuids']==result['gpu_identity']['uuids']==[low.UUID],'cleanup':receipt.get('cleanup_completed') is True and not receipt.get('cleanup_error'),'six_tasks':result['task_order']==job['tasks']==list(TASKS),'sample_count':all(t['sample_count']==4 for t in result['tasks']),'no_empty':all(t['empty_predictions']==0 for t in result['tasks']),'one_load':result['model_loads']==1,'engine_shutdown':result['engine_shutdown'] is True,'seven_artifacts':len(job['artifacts'])==7,'adapter_mode':bool(request['adapter_path'])==(name=='multitask'),'launch_arguments':execution['launches'][0]['command'][1]==str(REPO/'scripts/experiments/run_infer_job.py'),'launch_matches_job':execution['launches'][0]['pid']==receipt['pid']}.items():check(name+':'+key,passed)
            pids.update(receipt[k] for k in ('pid','launcher_pid') if k in receipt)
            for spec,expected in zip(request['tasks'],result['tasks']):
                observed=low.inspect_predictions(spec,low.load_samples(spec['annotation_path'],request['media_root']),request['params']); check(name+':prediction:'+spec['task'],observed==expected); predictions+=observed['sample_count']
            for entry_artifact in job['artifacts']:
                manifest=check_manifest(read_json(directory/entry_artifact['path']),verify_contents=True); check(name+':artifact:'+entry_artifact['role'],manifest['fingerprint']==entry_artifact['fingerprint'])
            for role,manifest in job['inputs'].items():check_manifest(manifest,verify_contents=True); check(name+':input:'+role,True)
            if name=='multitask':
                source=Path(request['adapter_path']); adoption=read_json(exp_dir/'stage-runs/tune/tune-smoke/jobs/tune-whole/adoption-evidence.json')
                check(name+':adopted_real_source_unchanged',sha256_file(source/'adapter_model.safetensors')==adoption['source_weight_sha256']=='29fc4abb3b00ce472944326c32df61a8c3f26a2a523026a9dcf3e4d50be37d80'); check(name+':no_training_claim',adoption['training_executed'] is False)
            log=(out/'logs/infer.log').read_text(errors='replace')
            check(name+':one_engine_log',log.count('Initializing a V1 LLM engine')==1); check(name+':one_model_load_log',log.count('Model loading took')==1); check(name+':no_traceback_oom','Traceback (most recent call last)' not in log and 'CUDA out of memory' not in log)
            warning_lines=[line for line in log.splitlines() if 'destroy_process_group' in line or 'Warning' in line or 'WARNING' in line]; warnings.extend({'setting':name,'line':line} for line in warning_lines)
        for line in (root/(name+'-gpu.jsonl')).read_text().splitlines():
            observation=json.loads(line)
            for row in observation.get('processes',[]):pids.add(int(row[0])); check(name+':sampled_gpu_uuid:'+str(row[0]),row[1]==low.UUID)
        totals.append({'setting':name,'predictions':predictions,'jobs':len(summary['jobs'])})
    for item in read_json(root/'samples.json'):check('original_annotation:'+item['task'],sha256_file(item['source'])==item['source_sha256'])
    for pid in sorted(pids):check('pid_exited:'+str(pid),not Path('/proc/'+str(pid)).exists())
    final=low.gpu_query(); write_json_atomic(root/'gpu-after.json',final); check('gpu_1_idle',not final['processes']); check('gpu_1_memory_released',float(final['gpu'][2])<50)
    summary={'successful':all(checks.values()),'checks_count':len(checks),'checks':checks,'pids':sorted(pids),'cases':totals,'exits':read_json(root/'gpu-exits.json'),'warnings':warnings,'training_executed':False,'scope':'4B physical GPU 1 high-level infer; no evaluation or quality assertion'}
    write_json_atomic(root/'verification.json',summary); print(json.dumps({'successful':summary['successful'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'cases':totals},ensure_ascii=False,indent=2)); return summary['successful']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--result-root',required=True,type=Path); p.add_argument('--audit-only',action='store_true'); a=p.parse_args(); root=a.result_root.resolve()
    if not root.is_relative_to((REPO/'smoke/results').resolve()):p.error('result-root must be under smoke/results')
    if not a.audit_only:low.run(root,setup(root))
    return 0 if audit(root) else 1
if __name__=='__main__':raise SystemExit(main())
