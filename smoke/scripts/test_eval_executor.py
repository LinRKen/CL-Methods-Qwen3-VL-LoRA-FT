'''CPU acceptance for batch eval. Payloads are explicit fakes; no model libraries or CUDA.'''
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
LAUNCH_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
SELF=Path(__file__).resolve()
sys.path.insert(0,str(SELF.parent))
_saved=sys.argv[:];sys.argv=[str(SELF)]
import test_infer_orchestrator as hi
sys.argv=_saved
from lib import eval_executor as ex, eval_worker as worker
import run_eval_job
from lib.common import TASKS,read_json,write_json_atomic,fingerprint,sha256_file
from lib.records import load_job,load_stage_run,record_status,bind_inputs
from lib.artifacts import make_manifest,check_manifest
RESULT=None
REAL_POPEN=hi.REAL_POPEN
REAL_LOCK=ex._exclusive

def fake_worker(path):
    req=read_json(path);legacy=worker._legacy(req['repo_root']);out=Path(req['output_dir']);events=[]
    mode=os.environ.get('SMOKE_EVAL_MODE','success')
    def event(value):
        events.append(value);write_json_atomic(out/'cpu-events.json',events)
    class Processor:
        def apply_chat_template(self,messages,**kw):return json.dumps(messages,ensure_ascii=False)
        def tokenizer(self,prompt,**kw):return {'input_ids':list(range(9000 if mode=='longprompt' else 10))}
    class Engine:
        def generate(self,prompts,**kw):
            event({'batch':len(prompts)})
            if mode=='generate':raise RuntimeError('intentional fake generation failure')
            outputs=[]
            for prompt in prompts:
                messages=json.loads(prompt);payload=json.loads(messages[1]['content'].rsplit('\n',1)[1]);q=payload['question']
                response={'status':'valid','score':0 if 'case=fail' in q else 1,'gt':'a','pred':'b'}
                if payload['metric_type']=='MRA':response={'status':'valid','score':None,'gt':10.,'pred':12.}
                if 'case=invalid' in q:response={'status':'invalid','score':None,'gt':None,'pred':None}
                content=json.dumps(response)
                if 'case=parsefail' in q or ('case=repair' in q and len(messages)==2):content='not-json'
                outputs.append(types.SimpleNamespace(finished=mode!='unfinished',outputs=[types.SimpleNamespace(text=content,finish_reason='abort' if mode=='abort' else 'stop')]))
            return outputs[:-1] if mode=='shortbatch' else outputs
    def load(*args):event('load-engine');return Engine(),object()
    def shutdown(engine):
        event('shutdown')
        if mode=='shutdown':raise RuntimeError('intentional fake shutdown failure')
    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':LAUNCH_CUDA,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}),patch.object(legacy,'verify_gpu_selection',return_value=worker._devices(req)),patch.object(legacy,'load_processor',side_effect=lambda *args:(event('load-processor') or Processor())),patch.object(legacy,'load_engine',side_effect=load),patch.object(legacy,'shutdown_engine',side_effect=shutdown):
        return worker.main(['--request',str(path)])

def execute(directory,inputs,mode='success'):
    directory=Path(directory)
    def lock(path,**kw):
        path=Path(path)
        if path.parent==Path('/tmp/qwen3vl-lora-tune-locks'):path=directory.parents[5]/'cpu-resource-locks'/path.name
        return REAL_LOCK(path,**kw)
    def spawn(command,**kw):
        env={**kw['env'],'SMOKE_EVAL_MODE':mode}
        return REAL_POPEN([sys.executable,'-B',str(SELF),'--fake-eval',command[-1]],**{**kw,'env':env})
    with patch.object(ex,'_exclusive',side_effect=lock),patch.object(ex,'_gpu_identity',side_effect=hi.identity),patch.object(ex.subprocess,'Popen',side_effect=spawn):
        return ex.execute_eval_job(directory,input_manifests=inputs)

class EvalTests(hi.T):
    def setUp(self):
        super().setUp()
        base=Path(self.doc['experiment']['base_model']);write_json_atomic(base/'config.json',{'model_type':'qwen3_vl'});(base/'model.safetensors').write_bytes(b'CPU FAKE WEIGHT; never loaded')
        self.doc['experiment']['judge_model']=str(base)
        self.doc['eval']={'run':{'python':sys.executable,'gpu_groups':[[1]],'max_parallel_jobs':1},'params':{'batch_size':2,'max_model_len':8192,'max_new_tokens':32,'repair_attempts':1}}
        for task in (*TASKS,'whole'):
            for split in ('train','test'):
                path=Path(self.doc['experiment']['dataset_root'])/task/(split+'.json');samples=read_json(path)
                for i,s in enumerate(samples):s['_meta']['metric_type']='OPEN';s['conversations'][0]['value']='题目 '+task+' case='+('pass' if i==0 else 'fail')
                write_json_atomic(path,samples)
    def ready(self,setting='zero_shot',extra=None):
        exp,runs,path=self.prepare(setting,extra);self.assertEqual(hi.T.execute(self,runs['infer'])['status'],'succeeded')
        directory=runs['eval']/'jobs/eval-judge-all'
        inputs={}
        for item in record_status(runs['infer'])['jobs']:
            source=runs['infer']/item['path']
            for entry in load_job(source)[0]['artifacts']:
                manifest=read_json(source/entry['path'])
                if manifest['kind']=='prediction':inputs['pred-'+str(len(inputs))]=manifest
        return directory,inputs,runs,path
    def run_eval(self,d,inputs,mode='success'):return execute(d,inputs,mode)
    def reject(self,d,inputs=None):
        with patch.object(ex,'_gpu_identity') as query,patch.object(ex.subprocess,'Popen') as spawn,self.assertRaises((ValueError,KeyError,OSError,RuntimeError,TypeError)):
            ex.execute_eval_job(d,input_manifests=inputs)
        query.assert_not_called();spawn.assert_not_called()
    def success(self,d,inputs,pairs=6,rows=12):
        receipt=self.run_eval(d,inputs);job,_=load_job(d);req=read_json(d/'evaluation.request.json');summary=worker.inspect_evaluation(req)
        self.assertEqual(receipt['status'],'succeeded');self.assertEqual(job['status'],'succeeded');self.assertEqual(summary['total'],rows);self.assertEqual(len(summary['pairs']),pairs);self.assertEqual(len(job['artifacts']),3*pairs+2)
        self.assertEqual(load_stage_run(d.parent.parent)[0]['status'],'running')
        events=read_json(Path(job['output_dir'])/'cpu-events.json');self.assertEqual(events.count('load-engine'),1);self.assertEqual(events.count('shutdown'),1)
        for e in job['artifacts']:check_manifest(read_json(d/e['path']),verify_contents=True)
        self.assertEqual(receipt['physical_gpu_ids'],[1]);self.assertTrue(receipt['cleanup_completed'])
        return req,summary
    def test_01_no_model_imports_and_compile(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','vllm','peft','wandb','deepspeed'})
        for f in ('run_eval_job.py','lib/eval_executor.py','lib/eval_worker.py'):compile((hi.REPO/'scripts/experiments'/f).read_text(),f,'exec')
    def test_02_zero_shot_all_six_pairs_one_engine(self):
        d,i,_,_=self.ready();req,s=self.success(d,i);self.assertEqual(s['score_sum'],6);self.assertEqual(s['combined_flat_rawacc'],.5);self.assertEqual([p['task'] for p in req['pairs']],list(TASKS))
    def test_03_multitask_same_adapter_all_tasks(self):
        d,i,_,_=self.ready('multitask');req,s=self.success(d,i);self.assertTrue(all(p['prediction_slot']=='whole' for p in req['pairs']))
    def test_04_singletask_six_sources(self):
        d,i,_,_=self.ready('singletask');req,s=self.success(d,i);self.assertEqual(len({p['prediction_job'] for p in req['pairs']}),6)
    def test_05_cl_all_36_pairs_no_deduplication(self):
        d,i,_,_=self.ready('cl');req,s=self.success(d,i,36,72);self.assertEqual(s['score_sum'],36);self.assertTrue(all(sum(p['task']==task for p in req['pairs'])==6 for task in TASKS))
    def test_06_transfer_round1_tests_source(self):
        d,i,r,_=self.ready('transfer_round1');req,s=self.success(d,i,30,60)
        for p in req['pairs']:
            src=load_job(r['infer']/'jobs'/p['prediction_job'])[0]['lineage']['adapter_ref'];self.assertEqual(p['task'],load_job(r['tune']/'jobs'/src)[0]['lineage']['source_task'])
    def test_07_transfer_round2_tests_target(self):
        d,i,r,_=self.ready('transfer_round2');req,s=self.success(d,i,30,60)
        for p in req['pairs']:
            src=load_job(r['infer']/'jobs'/p['prediction_job'])[0]['lineage']['adapter_ref'];self.assertEqual(p['task'],load_job(r['tune']/'jobs'/src)[0]['lineage']['train_task'])
    def test_08_semantics_repair_invalid_parsefail_mra_weighted(self):
        root=Path(self.doc['experiment']['dataset_root'])
        for task in TASKS:
            source=read_json(root/task/'test.json')[0];samples=[]
            modes=['pass','fail','mra','invalid','parsefail','repair'] if task=='motion' else ['pass']
            for n,mode in enumerate(modes):
                s=deepcopy(source);s['_meta'].update(id=task+str(n),sample_index=n+13,metric_type='MRA' if mode=='mra' else ('MCQ' if mode=='fail' else 'OPEN'));s['conversations'][0]['value']='题目 case='+mode;samples.append(s)
            write_json_atomic(root/task/'test.json',samples)
        d,i,_,_=self.ready();req,s=self.success(d,i,6,11)
        self.assertEqual(s['invalid'],2);self.assertEqual(s['judge_parse_failures'],1)
        legacy=worker._legacy(hi.REPO);expected=7.6
        self.assertAlmostEqual(s['score_sum'],expected);self.assertAlmostEqual(s['combined_flat_rawacc'],expected/11)
        raw=legacy.read_jsonl(Path(req['pairs'][0]['output_dir'])/'judge.raw.jsonl');self.assertEqual([len(x['attempts']) for x in raw],[1,1,1,1,2,2])
        self.assertIsNone(raw[5]['parse_error']);self.assertIsNotNone(raw[4]['parse_error']);self.assertEqual(raw[0]['record']['question'],'题目 case=pass')
    def test_09_missing_prediction_rejected(self):
        d,i,_,_=self.ready();i.pop(next(iter(i)));self.reject(d,i)
    def test_10_duplicate_prediction_rejected(self):
        d,i,_,_=self.ready();i['duplicate']=deepcopy(next(iter(i.values())));self.reject(d,i)
    def test_11_extra_input_role_rejected(self):
        d,i,_,_=self.ready();i['extra']=make_manifest('base_model',self.doc['experiment']['base_model'],source='smoke extra input');self.reject(d,i)
    def test_12_other_experiment_predictions_rejected(self):
        d,i,_,_=self.ready();_,foreign,_,_=self.ready();i[next(iter(i))]=next(iter(foreign.values()));self.reject(d,i)
    def test_13_non_successful_infer_run_rejected(self):
        d,i,r,_=self.ready();p=r['infer']/'run.json';doc=read_json(p);doc['status']='running';write_json_atomic(p,doc);self.reject(d,i)
    def test_14_non_successful_infer_job_rejected(self):
        d,i,r,_=self.ready();p=self.first(r['infer'])/'job.json';doc=read_json(p);doc['status']='failed';write_json_atomic(p,doc);self.reject(d,i)
    def test_15_changed_prediction_payload_rejected(self):
        d,i,_,_=self.ready();Path(next(iter(i.values()))['path']).write_text('tampered');self.reject(d,i)
    def test_16_changed_annotation_rejected(self):
        d,i,_,_=self.ready();p=Path(self.doc['experiment']['dataset_root'])/'motion/test.json';doc=read_json(p);doc[0]['conversations'][1]['value']='changed';write_json_atomic(p,doc);self.reject(d,i)
    def test_17_plan_pair_order_tamper_rejected(self):
        d,i,_,_=self.ready();p=d/'job.json';doc=read_json(p);doc['lineage']['scoring_pairs'].reverse();write_json_atomic(p,doc);self.reject(d,i)
    def test_18_job_tasks_tamper_rejected(self):
        d,i,_,_=self.ready();p=d/'job.json';doc=read_json(p);doc['tasks']=['geo'];write_json_atomic(p,doc);self.reject(d,i)
    def test_19_gpu_zero_config_rejected_before_query(self):
        with self.assertRaises(ValueError):self.prepare(extra={'eval':{'run':{'gpu_groups':[[0]]}}})
    def test_20_invalid_gpu_group_index_rejected(self):
        d,i,_,_=self.ready()
        for n in (-1,9,True):
            with patch.object(ex,'_gpu_identity') as q,self.assertRaises(ValueError):ex.execute_eval_job(d,gpu_group_index=n,input_manifests=i)
            q.assert_not_called()
    def test_21_existing_output_preserved(self):
        d,i,_,_=self.ready();out=Path(load_job(d)[0]['output_dir']);out.mkdir(parents=True);(out/'sentinel').write_text('keep');self.reject(d,i);self.assertEqual((out/'sentinel').read_text(),'keep')
    def test_22_existing_receipt_preserved(self):
        d,i,_,_=self.ready();p=d/'execution.json';p.write_text('keep');self.reject(d,i);self.assertEqual(p.read_text(),'keep')
    def test_23_bound_inputs_cannot_override(self):
        d,i,_,_=self.ready();j,s,exp,_=ex._context(d);ex._inputs(d,j,s,i,exp);self.reject(d,i)
    def test_24_terminal_job_not_restarted(self):
        d,i,_,_=self.ready();self.success(d,i);old=(d/'execution.json').read_bytes();self.reject(d,i);self.assertEqual((d/'execution.json').read_bytes(),old)
    def test_25_generate_failure_cleans_engine(self):self.fail_worker('generate')
    def test_26_shutdown_failure_fails_job(self):self.fail_worker('shutdown')
    def test_27_unfinished_generation_rejected(self):self.fail_worker('unfinished')
    def test_28_short_batch_rejected(self):self.fail_worker('shortbatch')
    def test_29_abort_generation_rejected(self):self.fail_worker('abort')
    def fail_worker(self,mode):
        d,i,_,_=self.ready()
        with self.assertRaises(RuntimeError):self.run_eval(d,i,mode)
        self.assertEqual(load_job(d)[0]['status'],'failed');receipt=read_json(d/'execution.json');self.assertTrue(receipt['cleanup_completed']);self.assertEqual(receipt['status'],'failed')
        events=read_json(Path(load_job(d)[0]['output_dir'])/'cpu-events.json');self.assertEqual(events.count('shutdown'),1);self.assertFalse(load_job(d)[0]['artifacts'])
    def test_30_long_prompt_fails_before_engine(self):
        d,i,_,_=self.ready()
        with self.assertRaises(RuntimeError):self.run_eval(d,i,'longprompt')
        events=read_json(Path(load_job(d)[0]['output_dir'])/'cpu-events.json');self.assertNotIn('load-engine',events)
    def test_31_raw_tamper_rejected(self):self.corrupt('raw')
    def test_32_scored_tamper_rejected(self):self.corrupt('scored')
    def test_33_pair_metrics_tamper_rejected(self):self.corrupt('pairmetrics')
    def test_34_combined_metrics_tamper_rejected(self):self.corrupt('metrics')
    def test_35_summary_tamper_rejected(self):self.corrupt('summary')
    def corrupt(self,kind):
        d,i,_,_=self.ready();req,s=self.success(d,i);pair=Path(req['pairs'][0]['output_dir']);out=Path(req['output_dir'])
        file={'raw':pair/'judge.raw.jsonl','scored':pair/'judge.scored.jsonl','pairmetrics':pair/'judge.metrics.json','metrics':out/'judge.metrics.json','summary':out/'evaluation_result.json'}[kind]
        if kind in ('raw','scored'):
            rows=[json.loads(x) for x in file.read_text().splitlines()];rows[0]['parsed' if kind=='raw' else 'sample_score']=None;file.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        else:
            doc=read_json(file);doc['score_sum']+=1;write_json_atomic(file,doc)
        with self.assertRaises((ValueError,TypeError)):worker.inspect_evaluation(req)
    def test_36_request_tamper_rejected(self):
        d,i,_,_=self.ready();req,_=self.success(d,i);req['params']['seed']+=1;p=self.case/'bad-request.json';write_json_atomic(p,req)
        with self.assertRaisesRegex(ValueError,'fingerprint'):worker._request(p)
    def test_37_gpu0_worker_rejected_before_query(self):
        d,i,_,_=self.ready();req,_=self.success(d,i);req['physical_gpu_ids']=[0];req['request_fingerprint']=fingerprint({k:v for k,v in req.items() if k!='request_fingerprint'});p=self.case/'bad-request.json';write_json_atomic(p,req)
        legacy=worker._legacy(hi.REPO)
        with patch.object(legacy,'verify_gpu_selection') as q,self.assertRaisesRegex(ValueError,'physical GPUs'):worker._request(p)
        q.assert_not_called()
    def test_38_cli_invalid_inputs_no_engine(self):
        for args in (['--job-dir','relative'],['--job-dir',str(self.case),'--input','a=relative'],['--job-dir',str(self.case),'--input','missing']):
            with patch.object(run_eval_job,'execute_eval_job') as launch:self.assertEqual(run_eval_job.main(args),1)
            launch.assert_not_called()
    def test_39_cli_success_failure_interrupt_codes(self):
        with patch.object(run_eval_job,'execute_eval_job',return_value={'status':'succeeded'}):self.assertEqual(run_eval_job.main(['--job-dir',str(self.case)]),0)
        with patch.object(run_eval_job,'execute_eval_job',side_effect=RuntimeError('smoke')):self.assertEqual(run_eval_job.main(['--job-dir',str(self.case)]),1)
        with patch.object(run_eval_job,'execute_eval_job',side_effect=KeyboardInterrupt()):self.assertEqual(run_eval_job.main(['--job-dir',str(self.case)]),130)
    def test_40_job_lock_rejects_duplicate(self):
        d,i,_,_=self.ready()
        with REAL_LOCK(d/'.execution.lock'):self.reject(d,i)
        self.assertFalse((d/'execution.json').exists())

def main():
    global RESULT
    p=argparse.ArgumentParser();p.add_argument('--result-root',required=True,type=Path);p.add_argument('--test',action='append',default=[]);a=p.parse_args();RESULT=a.result_root.resolve();RESULT.mkdir(parents=True,exist_ok=False)
    if not RESULT.is_relative_to((hi.REPO/'smoke/results').resolve()):p.error('smoke results only')
    hi.RESULT=RESULT
    names=a.test or [n for n in sorted(EvalTests.__dict__) if n.startswith('test_')];suite=unittest.TestSuite(EvalTests(n) for n in names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'details':[{'test':str(t),'traceback':s} for t,s in result.failures+result.errors]};write_json_atomic(RESULT/'summary.json',summary)
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fake-eval':raise SystemExit(fake_worker(Path(sys.argv[2])))
    raise SystemExit(main())

