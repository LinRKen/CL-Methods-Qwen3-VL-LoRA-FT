'''Adopt prior real 8B outputs using the production high-level CLI, without GPU work.'''
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/experiments'))
from lib.common import TASKS,read_json,write_json_atomic,sha256_file

def snapshot(roots,files):
    paths=set(files)
    for root in roots:paths.update(p for p in root.rglob('*') if p.is_file())
    return {str(p):{'sha256':sha256_file(p),'mode':p.stat().st_mode} for p in sorted(paths)}

def audit(s,run,expected,invalid):
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok:raise AssertionError(name)
    inv=Path(s['invocation_dir']);r=read_json(inv/'results.json');m=read_json(inv/'matrix.json');receipt=read_json(inv/'execution.json');rows=r['rows']
    check('run succeeded',s['status']==read_json(run/'run.json')['status']=='succeeded')
    check('experiment completed',read_json(run.parents[2]/'experiment.json')['status']=='succeeded')
    check('successful low job adopted',s['job']['reused_successful_job'] is True)
    check('no low child launched',receipt['launches']==[])
    check('one historical model load',r['model_loads']==1)
    check('six pairs',len(rows)==s['pair_count']==6)
    check('24 samples retained',s['overall']['total']==24)
    check('score sum',abs(s['overall']['score_sum']-expected)<1e-10)
    check('invalid retained',s['overall']['invalid']==invalid)
    check('no parse failures',s['overall']['judge_parse_failures']==0)
    check('weighted rawacc',abs(s['combined_flat_rawacc']-expected/24)<1e-10)
    check('task order',[x['test_task'] for x in rows]==list(TASKS))
    check('matrix agreement',r['matrix']==m==s['matrix'])
    check('overall agreement',r['overall']==s['overall'])
    check('matrix shape',len(m['values'])==1 and len(m['values'][0])==6)
    check('matrix values',m['values'][0]==[x['combined_flat_rawacc'] for x in rows])
    check('matrix counts',m['sample_counts'][0]==[4]*6)
    csvrows=list(csv.DictReader((inv/'results.csv').open()));mcsv=list(csv.reader((inv/'matrix.csv').open()))
    check('CSV order',[x['pair_id'] for x in csvrows]==r['pair_order'])
    check('matrix CSV axes',mcsv[0][1:]==list(TASKS) and mcsv[1][0]==m['row_order'][0])
    check('matrix CSV values',[float(v) for v in mcsv[1][1:]]==m['values'][0])
    for name,meta in s['result_files'].items():check('export SHA '+name,sha256_file(Path(meta['path']))==meta['sha256'])
    score=0;count=0
    for row,crow in zip(rows,csvrows):
        d=Path(row['output_dir']);scored=[json.loads(x) for x in (d/'judge.scored.jsonl').read_text().splitlines()];raw=[json.loads(x) for x in (d/'judge.raw.jsonl').read_text().splitlines()];v=sum(x['sample_score'] for x in scored);score+=v;count+=sum(x['status']!='valid' for x in scored);task=row['test_task']
        check(task+' raw/scored',len(scored)==len(raw)==row['total']==4)
        check(task+' score',abs(v-row['score_sum'])<1e-10 and abs(v/4-row['combined_flat_rawacc'])<1e-10)
        check(task+' CSV numbers',int(crow['total'])==4 and abs(float(crow['score_sum'])-v)<1e-10 and abs(float(crow['combined_flat_rawacc'])-v/4)<1e-10)
        check(task+' source files',all(Path(row[k]).is_file() for k in ['prediction_manifest','annotation_path','metrics_manifest']))
        pred=read_json(row['prediction_manifest']);metric=read_json(row['metrics_manifest'])
        check(task+' producer/hash',pred['fingerprint']==row['prediction_fingerprint'] and pred['producer']==row['prediction_producer'] and metric['fingerprint']==row['metrics_fingerprint'])
        check(task+' adapter',row['adapter'] is None and row['adapter_manifest'] is None if r['setting']=='zero_shot' else row['train_task']=='whole' and Path(row['adapter_manifest']).is_file())
    check('independent score',abs(score-expected)<1e-10);check('independent invalid',count==invalid)
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--result-root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.result_root.resolve()
    if not root.is_relative_to((REPO/'smoke/results').resolve()):p.error('smoke results only')
    root.mkdir(parents=True,exist_ok=a.resume);cases=[];base=REPO/'smoke/results/infer-orchestrator-20261028-t1qglfpo/gpu-real/records_root'
    for setting,score,invalid in [('zero_shot',5.9,14),('multitask',7.9,12)]:
        exp=(base/('smoke-'+setting)).resolve();run=exp/'stage-runs/eval/eval-smoke';job=run/'jobs/eval-judge-all';doc=read_json(job/'job.json');infer=exp/'stage-runs/infer/infer-smoke';irec=read_json(infer/'run.json')
        if doc['status']!='succeeded':raise ValueError('Expected existing successful low job')
        roots=[job,Path(doc['output_dir']),infer,Path(irec['output_root'])];files=[Path(v['path']) for v in doc['inputs'].values() if v['kind']=='annotation'];beforefile=root/(setting+'-before.json')
        if beforefile.exists() and a.resume:before=read_json(beforefile)
        else:
            before=snapshot(roots,files);write_json_atomic(beforefile,before)
        log=root/(setting+'-cli.log');command=[sys.executable,'-B',str(REPO/'scripts/experiments/run_eval.py'),'--run-dir',str(run)];started=time.monotonic()
        invocations=sorted((run/'orchestrations').glob('*/execution.json'))
        recovered=bool(invocations)
        if recovered:
            if not a.resume or len(invocations)!=1:raise ValueError('Unexpected prior invocation')
            receipt=read_json(invocations[0]);pid=receipt['pid'];code=None
            while Path('/proc/'+str(pid)).exists():
                state=Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()[0]
                if state=='Z':break
                if str(run).encode() not in Path('/proc/'+str(pid)+'/cmdline').read_bytes():raise ValueError('PID identity changed')
                time.sleep(1)
        else:
            if read_json(run/'run.json')['status']!='running':raise ValueError('High run not running')
            with log.open('xb') as f:
                process=subprocess.Popen(command,cwd=REPO,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GIT_OPTIONAL_LOCKS':'0'},stdout=f,stderr=subprocess.STDOUT);pid=process.pid
                write_json_atomic(root/(setting+'-launch.json'),{'pid':pid,'command':command,'started':time.time()})
                code=process.wait()
        item={'setting':setting,'run_dir':str(run),'pid':pid,'command':command,'returncode':code,'elapsed_seconds':time.monotonic()-started,'fresh_model_loads':0,'recovered_after_wrapper_timeout':recovered};cases.append(item);write_json_atomic(root/'cases.json',cases)
        if code not in (None,0):raise RuntimeError(str(log))
        s=read_json(run/'eval-summary.json');checks=audit(s,run,score,invalid);after=snapshot(roots,files)
        alive=Path('/proc/'+str(pid)).exists() and Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()[0]!='Z'
        checks.extend([{'name':'low/infer/input bytes and modes unchanged','passed':before==after},{'name':'CLI process no longer running','passed':not alive}]);item.update(checks=checks,passed=all(c['passed'] for c in checks),summary=s,protected_files=len(before));write_json_atomic(root/(setting+'-after.json'),after);write_json_atomic(root/'cases.json',cases)
        if not item['passed']:raise AssertionError(setting)
        print(json.dumps({k:item[k] for k in ['setting','returncode','elapsed_seconds','protected_files','passed']},ensure_ascii=False),flush=True)
    write_json_atomic(root/'audit.json',{'passed':True,'checks':sum(len(c['checks']) for c in cases),'cases':cases,'limitation':'Prior real 8B results adopted; no fresh judge initialization or GPU computation.'})
if __name__=='__main__':main()
