#!/usr/bin/env python3
"""Verify controlled interruption, full CLI resume, artifacts and CPU-only rebuilding."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'evaluation/COSINE'))
from dataset_utils import read_jsonl


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',required=True); opts=parser.parse_args()
    output=Path(opts.output_dir); output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'smoke/configs/eval_qwen_4b.json').read_text())
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               EVAL_SMOKE_ENV_AUDIT=str(output/'worker_environment.jsonl'))
    env['PYTHONPATH']=str(ROOT/'smoke/scripts/eval_qwen_audit')+os.pathsep+str(ROOT/'qwen-vl-finetune')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    model_args=['--dataset',config['dataset'],'--input-file',config['input_file'],
                '--output-dir',str(output),'--judge-model',config['judge_model']]
    observations=[]
    def run_phase(name,command,expected,phase_env):
        started=time.monotonic()
        with (output/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),env=phase_env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if time.monotonic()-started>900:
                        process.terminate()
                        raise TimeoutError(name+' exceeded 900 seconds')
                    memory=subprocess.run(['nvidia-smi','-i','1','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    apps=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    row={'phase':name,'at':datetime.now(timezone.utc).isoformat(),'memory':memory.stdout.strip(),'apps':apps.stdout.strip()}
                    observations.append(row)
                    with (output/'gpu_observations.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                    time.sleep(1)
                code=process.returncode
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=45)
        result={'phase':name,'exit_code':code,'expected_exit_code':expected,'seconds':time.monotonic()-started}
        print(json.dumps(result),flush=True)
        if code!=expected:
            print((output/(name+'.log')).read_text()[-5000:],flush=True)
            raise RuntimeError(name+' returned unexpected exit code')
        return result
    phases=[]
    phases.append(run_phase('interrupt',[sys.executable,str(ROOT/'smoke/scripts/interrupt_eval_qwen.py'),*model_args],75,env))
    raw=output/'judge.raw.jsonl'; prefix=raw.read_bytes(); first=read_jsonl(raw)
    assert len(first)==64, len(first)
    assert not (output/'judge.metrics.json').exists()
    (output/'interrupted.raw.jsonl').write_bytes(prefix)
    phases.append(run_phase('resume',[sys.executable,str(ROOT/'evaluation/COSINE/eval_cosine.py'),*model_args,'--resume'],0,env))
    assert raw.read_bytes().startswith(prefix)
    raw_hash=hashlib.sha256(raw.read_bytes()).hexdigest()
    scored_hash=hashlib.sha256((output/'judge.scored.jsonl').read_bytes()).hexdigest()
    metrics=json.loads((output/'judge.metrics.json').read_text())
    raw_rows=read_jsonl(raw); scored=read_jsonl(output/'judge.scored.jsonl')
    assert len(raw_rows)==len(scored)==metrics['total']==691
    assert sorted(r['record']['inference_index'] for r in raw_rows)==list(range(691))
    assert [r['inference_index'] for r in scored]==list(range(691))
    assert all(r['physical_gpus'][0]['physical_index']==1 and r['physical_gpus'][0]['uuid']==config['gpu_uuid'] for r in raw_rows)
    cpu_env=env.copy(); cpu_env['CUDA_VISIBLE_DEVICES']=''; cpu_env.pop('EVAL_SMOKE_ENV_AUDIT'); cpu_env['PYTHONPATH']=str(ROOT/'qwen-vl-finetune')
    env_shell=cpu_env.copy(); env_shell.update(DATASET=config['dataset'],INPUT_FILE=config['input_file'],OUTPUT_DIR=str(output),JUDGE_MODEL=config['judge_model'],EVAL_PYTHON=sys.executable)
    phases.append(run_phase('complete_resume',[sys.executable,str(ROOT/'evaluation/COSINE/eval_cosine.py'),*model_args,'--resume'],0,cpu_env))
    assert raw_hash==hashlib.sha256(raw.read_bytes()).hexdigest()
    assert scored_hash==hashlib.sha256((output/'judge.scored.jsonl').read_bytes()).hexdigest()
    rebuilt=json.loads((output/'judge.metrics.json').read_text())
    assert {k:v for k,v in metrics.items() if k!='completed_at'}=={k:v for k,v in rebuilt.items() if k!='completed_at'}
    assert 'Verified physical GPUs:' not in (output/'complete_resume.log').read_text()
    audits=read_jsonl(output/'worker_environment.jsonl')
    assert audits and all(r['CUDA_VISIBLE_DEVICES'] in ('1',config['gpu_uuid']) for r in audits)
    observed_pids=set()
    for row in observations:
        for app in row['apps'].splitlines():
            parts=[v.strip() for v in app.split(',')]
            if len(parts)==3:
                assert parts[0]==config['gpu_uuid']
                observed_pids.add(int(parts[1]))
    audited_pids={r['pid'] for r in audits}
    assert observed_pids and observed_pids<=audited_pids,(observed_pids,audited_pids)
    values=[float(r['memory'].split(',')[1].strip()) for r in observations if ',' in r['memory']]
    summary={'phases':phases,'interrupted_rows':len(first),'total':len(raw_rows),'prefix_preserved':True,
             'raw_sha256':raw_hash,'scored_sha256':scored_hash,'complete_resume_no_gpu':True,
             'worker_audit_rows':len(audits),'gpu_pids':sorted(observed_pids),'all_gpu_pids_audited':True,
             'observed_peak_gpu_memory_mib':max(values),'parse_failures':metrics['judge_parse_failures'],
             'repair_rows':sum(len(r['attempts'])>1 for r in raw_rows),'invalid':metrics['invalid'],
             'combined_flat_rawacc':metrics['combined_flat_rawacc'],'by_metric_type':metrics['by_metric_type']}
    (output/'integration.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
