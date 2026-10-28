"""Recompute the completed real eval evidence on CPU, without model imports."""
import argparse,fcntl,importlib.abc,json,os,re,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
class BlockModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','vllm','peft','wandb','deepspeed'}:raise ImportError('Model import forbidden in CPU audit: '+fullname)
sys.meta_path.insert(0,BlockModels())
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'scripts/experiments'))
from lib.common import TASKS,fingerprint,read_json,sha256_file,write_json_atomic
from lib.records import load_job,load_stage_run
from lib.artifacts import check_manifest
from lib.eval_worker import inspect_evaluation,inspect_pair

def main(root):
 checks={};pids=set();summaries=[];cpu_initial=read_json(root.parent/'cpu-initial/summary.json');cpu_retest=read_json(root.parent/'cpu-fixture-retest/summary.json');resource=read_json(root.parent/'cpu-resources/summary.json')
 expected_bad={'test_08_semantics_repair_invalid_parsefail_mra_weighted','test_11_extra_input_role_rejected'}
 checks['cpu_initial_only_two_fixture_errors']=cpu_initial['tests']==40 and cpu_initial['failures']==0 and cpu_initial['errors']==2 and all(any(n in x['test'] for n in expected_bad) for x in cpu_initial['details'])
 checks['cpu_fixture_retest_passed']=cpu_retest['tests']==2 and cpu_retest['successful']
 checks['resource_tests_passed']=resource['tests']==3 and resource['successful']
 for case in read_json(root/'cases.json'):
  name=case['setting'];prefix=name+':';d=Path(case['job_dir']);job,snapshot=load_job(d);req=read_json(d/'evaluation.request.json');receipt=read_json(d/'execution.json');out=Path(job['output_dir']);summary=inspect_evaluation(req);all_scored=[]
  checks[prefix+'cpu_full_reconstruction']=summary['status']=='succeeded'
  checks[prefix+'frozen_params']=req['params']==snapshot['config']['eval']['params']
  checks[prefix+'request_fingerprint']=req['request_fingerprint']==fingerprint({k:v for k,v in req.items() if k!='request_fingerprint'})
  checks[prefix+'request_copy']=read_json(out/'evaluation.request.json')==req
  checks[prefix+'receipt_copy']=read_json(out/'execution.json')==receipt
  checks[prefix+'cli_success']=read_json(root/(name+'-exit.json'))['returncode']==0
  checks[prefix+'worker_success_cleanup']=receipt['returncode']==0 and receipt['cleanup_completed'] and summary['engine_shutdown']
  checks[prefix+'one_judge_load']=summary['model_loads']==1
  checks[prefix+'gpu_environment']=receipt['environment']['CUDA_VISIBLE_DEVICES']=='1' and receipt['environment']['CUDA_DEVICE_ORDER']=='PCI_BUS_ID' and receipt['environment']['VLLM_WORKER_MULTIPROC_METHOD']=='spawn'
  checks[prefix+'task_order']=[p['task'] for p in req['pairs']]==list(TASKS)
  checks[prefix+'one_prediction_adapter']=len({p['prediction_job'] for p in req['pairs']})==1 and {p['prediction_slot'] for p in req['pairs']}==({'base'} if name=='zero_shot' else {'whole'})
  checks[prefix+'parent_running_is_expected']=load_stage_run(d.parent.parent)[0]['status']=='running'
  for pair in req['pairs']:
   key=prefix+pair['task']+':'
   pd=Path(pair['output_dir']);raw=[json.loads(x) for x in (pd/'judge.raw.jsonl').read_text().splitlines()];scored=[json.loads(x) for x in (pd/'judge.scored.jsonl').read_text().splitlines()];metrics=read_json(pd/'judge.metrics.json');result,rebuilt=inspect_pair(req,pair);all_scored.extend(scored)
   checks[key+'raw_complete']=len(raw)==4 and [x['record']['inference_index'] for x in raw]==list(range(4))
   checks[key+'scored_exact_reconstruction']=scored==rebuilt
   checks[key+'one_to_one_ids']=[(x['record']['id'],x['record']['sample_index']) for x in raw]==[(x['id'],x['sample_index']) for x in scored]
   checks[key+'nonempty_predictions']=all(isinstance(x['prediction'],str) and x['prediction'].strip() for x in scored)
   checks[key+'numeric_scores']=all(type(x['sample_score']) in (int,float) and 0<=x['sample_score']<=1 for x in scored)
   checks[key+'invalid_zero']=all(x['status']=='valid' or x['sample_score']==0 for x in scored)
   checks[key+'parse_fail_zero']=all(x['judge_parse_error'] is None or x['sample_score']==0 for x in scored)
   checks[key+'batch_and_repair_evidence']=all(1<=len(x['attempts'])<=req['params']['repair_attempts']+1 for x in raw)
   checks[key+'input_unchanged']=sha256_file(pair['prediction_path'])==pair['prediction_sha256'] and sha256_file(pair['annotation_path'])==pair['annotation_sha256']
   checks[key+'score_reduction']=metrics['total']==4 and abs(metrics['score_sum']-sum(x['sample_score'] for x in scored))<1e-10 and abs(metrics['combined_flat_rawacc']-metrics['score_sum']/4)<1e-10
   checks[key+'physical_device_evidence']=all(x['physical_gpus'][0]['physical_index']==1 and x['physical_gpus'][0]['uuid']==req['gpu_identity']['uuids'][0] for x in raw)
   checks[key+'registered_source']=check_manifest(read_json(root/(name+'-'+pair['prediction_role']+'.json')),verify_contents=True)['fingerprint']==pair['prediction_fingerprint']
   checks[key+'producer_success']=load_job(Path(case['experiment_dir'])/'stage-runs/infer'/pair['prediction_producer']['run_id']/'jobs'/pair['prediction_job'])[0]['status']=='succeeded'
   checks[key+'judge_text_only']=all(all(isinstance(m['content'],str) for m in a['messages']) for r in raw for a in r['attempts'])
  checks[prefix+'flat_denominator']=summary['total']==24 and abs(summary['score_sum']-sum(x['sample_score'] for x in all_scored))<1e-10 and abs(summary['combined_flat_rawacc']-sum(x['sample_score'] for x in all_scored)/24)<1e-10
  checks[prefix+'all_artifacts_registered']=len(job['artifacts'])==20 and len({e['role'] for e in job['artifacts']})==20 and all(check_manifest(read_json(d/e['path']),verify_contents=True)['fingerprint']==e['fingerprint'] for e in job['artifacts'])
  log=(out/'logs/eval.log').read_text(errors='replace');checks[prefix+'single_engine_log']=log.count('Initializing a V1 LLM engine')==1 and log.count('Model loading took')==1
  checks[prefix+'no_oom_traceback']=not any(x in log for x in ('Traceback (most recent call last)','CUDA out of memory','OutOfMemoryError'))
  for pid in re.findall(r'pid=(\d+)',log):pids.add(int(pid))
  pids.update(int(receipt[k]) for k in ('pid','launcher_pid'));pids.add(read_json(root/(name+'-launcher.json'))['pid'])
  for row in map(json.loads,(root/(name+'-gpu.jsonl')).read_text().splitlines()):
   for proc in row.get('processes',[]):pids.add(int(proc[0]))
  for path in [d/'.execution.lock',d.parent.parent/'.eval-slots'/str(receipt['stage_slot'])]:
   if path.name.isdigit():path=path.with_suffix('.lock')
   with path.open('r') as f:
    try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);checks[prefix+'lock_released:'+path.name]=True
    except BlockingIOError:checks[prefix+'lock_released:'+path.name]=False
  summaries.append({k:summary[k] for k in ('producer','total','valid','invalid','score_sum','combined_flat_rawacc','judge_parse_failures','pairs')})
 for pid in sorted(pids):checks['pid_exited:'+str(pid)]=not Path('/proc/'+str(pid)).exists()
 checks['no_model_imports']=not ({'torch','transformers','vllm','peft'} & set(sys.modules))
 report={'successful':all(checks.values()),'checks':checks,'pids':sorted(pids),'summaries':summaries};write_json_atomic(root/'audit-extended.json',report)
 print(json.dumps({'successful':report['successful'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'pids':sorted(pids),'summaries':[{k:v for k,v in x.items() if k!='pairs'} for x in summaries]},indent=2));return 0 if report['successful'] else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result-root',required=True,type=Path);a=p.parse_args();root=a.result_root.resolve()
 if not root.is_relative_to((REPO/'smoke/results').resolve()):p.error('smoke results only')
 raise SystemExit(main(root))
