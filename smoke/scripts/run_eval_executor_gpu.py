'''Execute the two frozen eval jobs from the completed infer smoke on physical GPU 1.'''
import argparse,csv,json,os,subprocess,sys,threading,time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'scripts/experiments'))
from lib.common import TASKS,read_json,write_json_atomic,sha256_file
from lib.records import load_job,load_stage_run,record_status
from lib.artifacts import check_manifest
from lib.eval_worker import inspect_evaluation
BASE=REPO/'smoke/results/infer-orchestrator-20261028-t1qglfpo/gpu-real'
ROOT=None;PYTHON='/home/linruikang/data/envs/Qwen3VL-vLLM-Verify-20261025/bin/python';UUID='GPU-63a25b73-5c01-919f-188a-29c116904574'
def gpu_query():
 def query(kind,fields):return list(csv.reader(subprocess.check_output(['nvidia-smi','-i','1','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines(),skipinitialspace=True))
 cards=query('gpu','index,uuid,memory.used,memory.total');procs=query('compute-apps','pid,gpu_uuid,used_memory')
 if len(cards)!=1 or cards[0][:2]!=['1',UUID] or any(row[1]!=UUID for row in procs):raise RuntimeError('Physical GPU identity mismatch: '+repr((cards,procs)))
 return {'time':time.time(),'gpu':cards[0],'processes':procs}
def setup():
 ROOT.mkdir(parents=True,exist_ok=False);cases=[]
 for setting in ('zero_shot','multitask'):
  exp=BASE/'records_root'/('smoke-'+setting);run=exp/'stage-runs/infer/infer-smoke';evrun=exp/'stage-runs/eval/eval-smoke';jobdir=evrun/'jobs/eval-judge-all'
  infer,_=load_stage_run(run);evjob,_=load_job(jobdir)
  if infer['status']!='succeeded' or record_status(jobdir.parent.parent)['status']!='planned' or evjob['status']!='planned':raise RuntimeError('Source infer/eval state is not the expected clean frozen state')
  bindings={}
  for entry in infer['jobs']:
   source=run/'jobs'/entry['job_id'];job,_=load_job(source)
   if job['status']!='succeeded':raise RuntimeError('Infer producer is not successful')
   for artifact in job['artifacts']:
    manifest=read_json(source/artifact['path'])
    if manifest['kind']=='prediction':
     check_manifest(manifest,verify_contents=True);role='prediction-'+str(len(bindings)).zfill(2);bindings[role]=manifest
  if len(bindings)!=6:raise RuntimeError('Expected six frozen prediction manifests')
  command=[PYTHON,'-B',str(REPO/'scripts/experiments/run_eval_job.py'),'--job-dir',str(jobdir),'--gpu-group-index','0']
  for role,manifest in bindings.items():
   file=ROOT/(setting+'-'+role+'.json');write_json_atomic(file,manifest);command+=['--input',role+'='+str(file)]
  cases.append({'setting':setting,'experiment_dir':str(exp),'run_dir':str(evrun),'job_dir':str(jobdir),'command':command,'prediction_count':len(bindings)})
 write_json_atomic(ROOT/'cases.json',cases);return cases
def run():
 before=gpu_query();write_json_atomic(ROOT/'gpu-before.json',before)
 if before['processes']:raise RuntimeError('Physical GPU 1 is occupied; no evaluation launched')
 cases=read_json(ROOT/'cases.json');results=[]
 for case in cases:
  observations=[];stop=threading.Event();start=time.monotonic();name=case['setting']
  def monitor():
   with (ROOT/(name+'-gpu.jsonl')).open('x') as f:
    while not stop.is_set():
     try:row=gpu_query()
     except Exception as e:row={'time':time.time(),'error':repr(e)}
     observations.append(row);f.write(json.dumps(row)+'\n');f.flush();stop.wait(2)
  log=(ROOT/(name+'-cli.log')).open('xb');env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONDONTWRITEBYTECODE':'1'}
  p=subprocess.Popen(case['command'],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write_json_atomic(ROOT/(name+'-launcher.json'),{'pid':p.pid,'command':case['command']});thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:code=p.wait()
  except BaseException:p.terminate();p.wait();raise
  finally:stop.set();thread.join(timeout=15);log.close()
  evidence={'setting':name,'returncode':code,'elapsed_seconds':time.monotonic()-start,'observations':len(observations),'observed_max_whole_gpu_mib':max((float(x['gpu'][2]) for x in observations if 'gpu'in x),default=None)};write_json_atomic(ROOT/(name+'-exit.json'),evidence);results.append(evidence);print(json.dumps(evidence),flush=True)
  if code:raise RuntimeError(name+' eval failed; preserving evidence and not retrying')
 write_json_atomic(ROOT/'gpu-exits.json',results)
def audit():
 checks={};pids=set();summaries=[]
 for case in read_json(ROOT/'cases.json'):
  samples=[json.loads(line) for line in (ROOT/(case['setting']+'-gpu.jsonl')).read_text().splitlines()]
  for row in samples:
   for proc in row.get('processes',[]):pids.add(int(proc[0]))
  name=case['setting'];d=Path(case['job_dir']);job,_=load_job(d);receipt=read_json(d/'execution.json');req=read_json(d/'evaluation.request.json');out=Path(job['output_dir']);summary=inspect_evaluation(req);summaries.append({'setting':name,'total':summary['total'],'valid':summary['valid'],'invalid':summary['invalid'],'score_sum':summary['score_sum'],'combined_flat_rawacc':summary['combined_flat_rawacc'],'pairs':summary['pairs']})
  checks[name+':job_succeeded']=job['status']=='succeeded';checks[name+':receipt_succeeded']=receipt['status']=='succeeded' and receipt['returncode']==0 and receipt['cleanup_completed'] is True
  checks[name+':gpu_identity']=receipt['physical_gpu_ids']==[1] and receipt['gpu_identity']['uuids']==[UUID]
  checks[name+':one_model_load']=summary['model_loads']==1 and '"model_loads": 1' in (out/'evaluation_result.json').read_text()
  checks[name+':all_tasks']=[x['task'] for x in summary['pairs']]==list(TASKS)
  checks[name+':six_pairs']=len(summary['pairs'])==6 and all(x['total']==4 for x in summary['pairs']) and summary['total']==24
  checks[name+':judge_configuration']=Path(req['judge_model']).name=='Qwen3-VL-8B-Instruct' and req['physical_gpu_ids']==[1] and req['params']['batch_size']==4
  checks[name+':gpu_samples']=bool(samples) and all('gpu' in row and not row.get('error') and row['gpu'][:2]==['1',UUID] for row in samples)
  checks[name+':artifacts']=len(job['artifacts'])==20 and all(check_manifest(read_json(d/e['path']),verify_contents=True)['fingerprint']==e['fingerprint'] for e in job['artifacts'])
  checks[name+':batch_size']=all(sum(1 for line in (Path(p['output_dir'])/'judge.raw.jsonl').read_text().splitlines())==4 for p in req['pairs'])
  checks[name+':input_hashes']=all(sha256_file(p['prediction_path'])==p['prediction_sha256'] and sha256_file(p['annotation_path'])==p['annotation_sha256'] for p in req['pairs'])
  log=(out/'logs/eval.log').read_text(errors='replace');checks[name+':engine_log_once']=log.count('Initializing a V1 LLM engine')==1 and log.count('Model loading took')==1
  pids.update(receipt[k] for k in ('pid','launcher_pid') if k in receipt);pids.add(read_json(ROOT/(name+'-launcher.json'))['pid'])
 final=gpu_query();checks['gpu1_idle']=not final['processes'];checks['gpu1_memory_released']=float(final['gpu'][2])<50
 for pid in sorted(pids):checks['pid_exited:'+str(pid)]=not Path('/proc/'+str(pid)).exists()
 write_json_atomic(ROOT/'gpu-after.json',final);write_json_atomic(ROOT/'verification.json',{'successful':all(checks.values()),'checks':checks,'summaries':summaries,'pids':sorted(pids)});print(json.dumps({'successful':all(checks.values()),'checks':checks,'summaries':summaries},indent=2));return all(checks.values())
def main():
 global ROOT
 p=argparse.ArgumentParser();p.add_argument('--result-root',required=True,type=Path);p.add_argument('mode',choices=['setup','run','audit']);a=p.parse_args();ROOT=a.result_root.resolve()
 if not ROOT.is_relative_to((REPO/'smoke/results').resolve()):p.error('result root must be below smoke/results')
 if a.mode=='setup':setup()
 elif a.mode=='run':run()
 else:raise SystemExit(0 if audit() else 1)
if __name__=='__main__':main()

