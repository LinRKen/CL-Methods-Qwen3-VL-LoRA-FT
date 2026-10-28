'''Step 9 CPU acceptance. Models/GPU identity are explicit fakes; production orchestration is real.'''
import argparse
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
LAUNCH_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
SELF=Path(__file__).resolve()
sys.path.insert(0,str(SELF.parent))
saved=sys.argv[:];sys.argv=[str(SELF)]
import test_eval_executor as fx
sys.argv=saved
hi=fx.hi
from lib import eval_orchestrator as orch
from lib.common import TASKS,read_json,write_json_atomic,sha256_file,fingerprint
from lib.records import load_job,load_stage_run,record_status,update_status
import run_eval
REAL_POPEN=fx.REAL_POPEN
RESULT=None

def fake_job(argv):
    p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--gpu-group-index',type=int,default=0);p.add_argument('--input',action='append',default=[]);a=p.parse_args(argv)
    d=a.job_dir;mode=os.environ.get('SMOKE_HIGH_EVAL_MODE','success')
    event={'pid':os.getpid(),'launch_cuda':LAUNCH_CUDA,'inputs':a.input,'started':time.time(),'mode':mode}
    write_json_atomic(d/'cpu-high-job.json',event)
    def lock(path,**kw):
        path=Path(path)
        if path.parent==Path('/tmp/qwen3vl-lora-tune-locks'):path=Path(load_job(d)[1]['config']['experiment']['records_root']).parent/'cpu-eval-locks'/path.name
        return fx.REAL_LOCK(path,**kw)
    def spawn(command,**kw):
        target=[sys.executable,'-B',str(SELF),'--slow-worker',command[-1]] if mode=='slow' else [sys.executable,'-B',str(fx.SELF),'--fake-eval',command[-1]]
        env={**kw['env'],'SMOKE_EVAL_MODE':mode if mode in ('generate','shutdown','shortbatch') else 'success'}
        return REAL_POPEN(target,**{**kw,'env':env})
    try:
        inputs={k:read_json(v) for k,v in (x.split('=',1) for x in a.input)}
        with patch.object(fx.ex,'_exclusive',side_effect=lock),patch.object(fx.ex,'_gpu_identity',side_effect=hi.identity),patch.object(fx.ex.subprocess,'Popen',side_effect=spawn):
            fx.ex.execute_eval_job(d,gpu_group_index=a.gpu_group_index,input_manifests=inputs)
        job,_=load_job(d);out=Path(job['output_dir'])
        if mode in ('receipt','cleanup'):
            receipt=read_json(d/'execution.json');receipt['config_fingerprint' if mode=='receipt' else 'cleanup_completed']='tampered' if mode=='receipt' else False
            write_json_atomic(d/'execution.json',receipt)
        elif mode=='request':
            req=read_json(d/'evaluation.request.json');req['params']['seed']+=1;write_json_atomic(d/'evaluation.request.json',req)
        elif mode=='raw':
            req=read_json(d/'evaluation.request.json');(Path(req['pairs'][0]['output_dir'])/'judge.raw.jsonl').write_text('{}\n')
        elif mode=='metrics':
            doc=read_json(out/'judge.metrics.json');doc['score_sum']+=1;write_json_atomic(out/'judge.metrics.json',doc)
        elif mode=='artifact':
            path=d/job['artifacts'][0]['path'];doc=read_json(path);doc['metadata']['sample_count']+=1;write_json_atomic(path,doc)
        return 0
    except BaseException as error:
        event['error']=repr(error);return 17
    finally:
        event.update(finished=time.time(),status=record_status(d)['status']);write_json_atomic(d/'cpu-high-job.json',event)

def dispatch(command,**kw):
    return REAL_POPEN([command[0],'-B',str(SELF),'--fake-job',*command[2:]],**kw)

class HighTests(fx.EvalTests):
    def high(self,run,mode='success'):
        with patch.dict(os.environ,{'SMOKE_HIGH_EVAL_MODE':mode}),patch.object(orch.subprocess,'Popen',side_effect=dispatch):
            return orch.execute_eval_run(run)
    def reject_high(self,run):
        before=set((run/'orchestrations').glob('*'))
        with patch.object(orch.subprocess,'Popen') as spawn,patch.object(fx.ex,'_gpu_identity') as query,self.assertRaises((ValueError,KeyError,OSError,TypeError)):
            orch.execute_eval_run(run)
        spawn.assert_not_called();query.assert_not_called();self.assertEqual(set((run/'orchestrations').glob('*')),before)
    def check_exports(self,s,run,pairs,expected_total):
        self.assertEqual(s['status'],'succeeded',s.get('details'));self.assertEqual(s['pair_count'],pairs);self.assertEqual(s['overall']['total'],expected_total)
        self.assertEqual(record_status(run)['status'],'succeeded');self.assertEqual(read_json(run/'eval-summary.json'),s)
        folder=Path(s['invocation_dir']);r=read_json(folder/'results.json');m=read_json(folder/'matrix.json')
        self.assertEqual(r['matrix'],m);self.assertEqual(r['overall'],s['overall']);self.assertEqual(len(r['rows']),pairs)
        rows=list(csv.DictReader((folder/'results.csv').open()));self.assertEqual([x['pair_id'] for x in rows],r['pair_order'])
        for name,meta in s['result_files'].items():self.assertEqual(meta['sha256'],sha256_file(Path(meta['path'])))
        total=sum(row['total'] for row in r['rows']);score=sum(row['score_sum'] for row in r['rows'])
        self.assertEqual(total,expected_total);self.assertAlmostEqual(r['overall']['score_sum'],score);self.assertAlmostEqual(r['overall']['combined_flat_rawacc'],score/total)
        cm=list(csv.reader((folder/'matrix.csv').open()));self.assertEqual(cm[0][1:],list(TASKS));self.assertEqual([x[0] for x in cm[1:]],m['row_order'])
        for ri,line in enumerate(m['values']):
            for ci,value in enumerate(line):self.assertEqual(cm[ri+1][ci+1],'' if value is None else str(value))
        receipt=read_json(folder/'execution.json');self.assertEqual(receipt['status'],'succeeded');self.assertEqual(receipt['job_order'],['eval-judge-all'])
        self.assertEqual(sum(x is not None for line in m['pair_ids'] for x in line),pairs)
        return r
    def standard(self,setting,pairs,extra=None):
        d,_,runs,_=self.ready(setting,extra);s=self.high(runs['eval']);r=self.check_exports(s,runs['eval'],pairs,2*pairs)
        receipt=read_json(Path(s['invocation_dir'])/'execution.json');self.assertEqual(len(receipt['launches']),1);self.assertEqual(len(receipt['input_manifests']),pairs)
        events=read_json(Path(load_job(d)[0]['output_dir'])/'cpu-events.json');self.assertEqual(events.count('load-engine'),1);self.assertEqual(events.count('shutdown'),1)
        self.assertEqual(record_status(runs['eval'].parents[2])['status'],'succeeded');self.assertEqual(read_json(d/'cpu-high-job.json')['launch_cuda'],'1')
        return d,r,runs,s
    def test_01_import_compile_without_model_libraries(self):
        self.assertFalse(set(sys.modules)&{'torch','transformers','vllm','peft','wandb','deepspeed'})
        for f in ('run_eval.py','lib/eval_orchestrator.py'):compile((hi.REPO/'scripts/experiments'/f).read_text(),f,'exec')
    def test_02_zero_shot_six_pairs_one_launch(self):self.standard('zero_shot',6)
    def test_03_multitask_six_pairs_one_adapter(self):
        _,r,_,_=self.standard('multitask',6);self.assertTrue(all(x['train_task']=='whole' and x['adapter_manifest'] for x in r['rows']))
    def test_04_singletask_diagonal_and_unmeasured_null(self):
        _,r,_,_=self.standard('singletask',6);m=r['matrix']
        for i in range(6):
            for j in range(6):self.assertEqual(m['values'][i][j],.5 if i==j else None)
    def test_05_cl_full_36_pairs_custom_order(self):
        order=list(reversed(TASKS));_,r,_,_=self.standard('cl',36,{'experiment':{'task_order':order}});self.assertEqual(r['matrix']['row_order'],order)
        for row in r['rows']:self.assertEqual(row['cl_stage'],order.index(row['train_task'])+1);self.assertEqual(row['matrix_column'],row['test_task'])
    def test_06_transfer_round1_source_task_and_30_cells(self):self.transfer('transfer_round1','source_task')
    def test_07_transfer_round2_target_task_and_30_cells(self):self.transfer('transfer_round2','second_round_task')
    def transfer(self,setting,key):
        _,r,_,_=self.standard(setting,30);self.assertEqual(r['matrix']['test_task_rule'],key)
        for row in r['rows']:self.assertEqual(row['test_task'],row[key]);self.assertEqual(row['matrix_row'],row['source_task']);self.assertEqual(row['matrix_column'],row['second_round_task'])
        self.assertTrue(all(r['matrix']['values'][i][i] is None for i in range(6)))
    def test_08_weighted_score_invalid_parsefail_repair_mra(self):
        root=Path(self.doc['experiment']['dataset_root'])
        for task in TASKS:
            source=read_json(root/task/'test.json')[0];rows=[]
            for n,mode in enumerate(['pass','fail','mra','invalid','parsefail','repair'] if task=='motion' else ['fail']):
                row=deepcopy(source);row['_meta'].update(id=task+str(n),sample_index=n,metric_type='MRA' if mode=='mra' else 'OPEN');row['conversations'][0]['value']='case='+mode;rows.append(row)
            write_json_atomic(root/task/'test.json',rows)
        _,_,runs,_=self.ready();s=self.high(runs['eval']);r=self.check_exports(s,runs['eval'],6,11)
        self.assertAlmostEqual(r['overall']['score_sum'],2.6);self.assertAlmostEqual(r['overall']['combined_flat_rawacc'],2.6/11);self.assertEqual(r['overall']['invalid'],2);self.assertEqual(r['overall']['judge_parse_failures'],1)
        self.assertNotAlmostEqual(r['overall']['combined_flat_rawacc'],sum(x['combined_flat_rawacc'] for x in r['rows'])/6)
        self.assertEqual(r['matrix']['values'][0][1],0)
    def test_09_adopt_successful_low_job_without_spawn(self):
        d,inputs,runs,_=self.ready();self.run_eval(d,inputs);before={str(p):sha256_file(p) for p in d.rglob('*') if p.is_file()}
        with patch.object(orch.subprocess,'Popen') as launch,patch.object(fx.ex,'_gpu_identity') as q:s=orch.execute_eval_run(runs['eval'])
        launch.assert_not_called();q.assert_not_called();self.check_exports(s,runs['eval'],6,12);self.assertTrue(s['job']['reused_successful_job']);self.assertEqual(read_json(Path(s['invocation_dir'])/'execution.json')['launches'],[])
        self.assertEqual(before,{str(p):sha256_file(p) for p in d.rglob('*') if p.is_file()})
    def test_10_frozen_configuration_survives_deleted_yaml(self):
        _,_,runs,path=self.ready();path.unlink();self.check_exports(self.high(runs['eval']),runs['eval'],6,12)
    def test_11_missing_infer_registration(self):self.registration('missing')
    def test_12_duplicate_infer_registration(self):self.registration('duplicate')
    def registration(self,mode):
        _,_,r,_=self.ready();p=r['eval'].parents[2]/'experiment.json';doc=read_json(p);doc['stage_runs']=[i for i in doc['stage_runs'] if i['stage']!='infer'] if mode=='missing' else doc['stage_runs']+[next(i for i in doc['stage_runs'] if i['stage']=='infer')];write_json_atomic(p,doc);self.reject_high(r['eval'])
    def test_13_unsuccessful_infer_run(self):self.mutate_reject('infer-run')
    def test_14_unsuccessful_infer_job(self):self.mutate_reject('infer-job')
    def test_15_changed_prediction(self):self.mutate_reject('prediction')
    def test_16_changed_ground_truth(self):self.mutate_reject('gt')
    def test_17_changed_infer_plan(self):self.mutate_reject('infer-plan')
    def test_18_changed_eval_pair_order(self):self.mutate_reject('eval-pairs')
    def test_19_changed_eval_tasks(self):self.mutate_reject('eval-tasks')
    def test_20_cross_experiment_infer(self):self.mutate_reject('foreign')
    def test_21_changed_registration_path(self):self.mutate_reject('registration-path')
    def mutate_reject(self,kind):
        d,_,r,_=self.ready()
        if kind in ('infer-run','foreign'):p=r['infer']/'run.json';doc=read_json(p);doc['status' if kind=='infer-run' else 'experiment_id']='running' if kind=='infer-run' else 'foreign'
        elif kind=='infer-job':p=self.first(r['infer'])/'job.json';doc=read_json(p);doc['status']='failed'
        elif kind=='prediction':p=Path(read_json(self.first(r['infer'])/'artifacts/prediction-motion.json')['path']);p.write_text('corrupt');doc=None
        elif kind=='gt':p=Path(self.doc['experiment']['dataset_root'])/'motion/test.json';doc=read_json(p);doc[0]['conversations'][1]['value']='changed'
        elif kind=='infer-plan':p=r['infer']/'plan.json';doc=read_json(p);doc['jobs'][0]['tasks'].reverse()
        elif kind in ('eval-pairs','eval-tasks'):p=d/'job.json';doc=read_json(p);doc['lineage']['scoring_pairs'].reverse() if kind=='eval-pairs' else doc.update(tasks=['motion'])
        else:p=r['eval'].parents[2]/'experiment.json';doc=read_json(p);next(i for i in doc['stage_runs'] if i['stage']=='infer')['path']='stage-runs/infer/wrong'
        if doc is not None:write_json_atomic(p,doc)
        self.reject_high(r['eval'])
    def test_22_planned_receipt_preserved(self):self.existing('receipt')
    def test_23_planned_request_preserved(self):self.existing('request')
    def test_24_planned_output_preserved(self):self.existing('output')
    def existing(self,kind):
        d,_,r,_=self.ready();p=d/('execution.json' if kind=='receipt' else 'evaluation.request.json') if kind!='output' else Path(load_job(d)[0]['output_dir'])/'sentinel';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('keep');self.reject_high(r['eval']);self.assertEqual(p.read_text(),'keep')
    def test_25_running_job_rejected(self):self.lifecycle('running')
    def test_26_failed_job_rejected(self):self.lifecycle('failed')
    def test_27_cancelled_job_rejected(self):self.lifecycle('cancelled')
    def lifecycle(self,status):
        d,_,r,_=self.ready();p=d/'job.json';doc=read_json(p);doc['status']=status;write_json_atomic(p,doc);self.reject_high(r['eval'])
    def test_28_terminal_high_run_not_restarted(self):
        _,_,r,_=self.ready();self.high(r['eval']);self.reject_high(r['eval'])
    def test_29_job_generate_failure_no_exports(self):self.failed('generate')
    def test_30_job_shutdown_failure_no_exports(self):self.failed('shutdown')
    def test_31_receipt_corruption_rejected(self):self.failed('receipt')
    def test_32_cleanup_corruption_rejected(self):self.failed('cleanup')
    def test_33_request_corruption_rejected(self):self.failed('request')
    def test_34_raw_corruption_rejected(self):self.failed('raw')
    def test_35_metrics_corruption_rejected(self):self.failed('metrics')
    def test_36_artifact_corruption_rejected(self):self.failed('artifact')
    def failed(self,mode):
        d,_,r,_=self.ready();s=self.high(r['eval'],mode);self.assertEqual(s['status'],'failed');self.assertFalse(s['result_files']);self.assertTrue(s['details']);folder=Path(s['invocation_dir']);self.assertTrue((d/'execution.json').exists())
        self.assertFalse(any((folder/n).exists() for n in ('results.json','results.csv','matrix.json','matrix.csv')))
        self.assertEqual(record_status(r['eval'])['status'],'failed');self.assertEqual(record_status(r['eval'].parents[2])['status'],'active')
    def test_37_high_lock_rejects_second_launcher(self):
        _,_,r,_=self.ready()
        with orch._run_lock(r['eval']):self.reject_high(r['eval'])
    def test_38_launch_exception_recorded(self):
        _,_,r,_=self.ready()
        with patch.object(orch.subprocess,'Popen',side_effect=OSError('intentional spawn failure')):s=orch.execute_eval_run(r['eval'])
        self.assertEqual(s['status'],'failed');self.assertEqual(read_json(Path(s['invocation_dir'])/'execution.json')['launches'][0]['status'],'launch_failed');self.assertFalse(s['result_files'])
    def test_39_partial_publication_failure_removes_exports(self):
        _,_,r,_=self.ready();original=orch._write_text_atomic
        def write(path,content):
            if path.name=='matrix.csv':raise OSError('intentional publish failure')
            return original(path,content)
        with patch.object(orch,'_write_text_atomic',side_effect=write):s=self.high(r['eval'])
        self.assertEqual(s['status'],'failed');self.assertFalse(any((Path(s['invocation_dir'])/n).exists() for n in ('results.json','results.csv','matrix.json','matrix.csv')))
    def test_40_cli_codes_and_relative_path(self):
        with patch.object(run_eval,'execute_eval_run') as fn:self.assertEqual(run_eval.main(['--run-dir','relative']),1);fn.assert_not_called()
        for status,code in [('succeeded',0),('failed',1),('cancelled',130)]:
            with patch.object(run_eval,'execute_eval_run',return_value={'status':status}):self.assertEqual(run_eval.main(['--run-dir',str(self.case)]),code)
        with patch.object(run_eval,'execute_eval_run',side_effect=KeyboardInterrupt()):self.assertEqual(run_eval.main(['--run-dir',str(self.case)]),130)
    def test_41_resources_forbid_gpu_zero_and_wrong_tp(self):
        for group,tp in [([0],1),([1,1],2),([1],2),([True],1),([],1)]:
            with self.assertRaises(ValueError):orch._resources({'eval':{'run':{'gpu_groups':[group]},'params':{'tensor_parallel_size':tp}}})
    def test_42_first_gpu_group_only_one_judge(self):
        _,_,r,_=self.ready(extra={'eval':{'run':{'gpu_groups':[[1],[2]],'max_parallel_jobs':2}}});s=self.high(r['eval']);self.check_exports(s,r['eval'],6,12);receipt=read_json(Path(s['invocation_dir'])/'execution.json');self.assertEqual(len(receipt['launches']),1);self.assertEqual(receipt['launches'][0]['assignment']['physical_gpu_ids'],[1])
    def test_43_matrix_rejects_duplicates_missing_and_invalid(self):
        _,r,_,_=self.standard('zero_shot',6);cfg={'setting':'zero_shot','task_order':list(TASKS)}
        for rows in (r['rows'][:-1],r['rows']+[r['rows'][0]],[{**row,'combined_flat_rawacc':float('nan')} for row in r['rows']],[{**row,'combined_flat_rawacc':1.1} for row in r['rows']]):
            with self.assertRaises(ValueError):orch._matrix(rows,cfg)
    def test_44_sigterm_propagates_and_waits_for_worker_cleanup(self):
        d,_,r,_=self.ready();log=(self.case/'interrupt.log').open('xb');env={**os.environ,'SMOKE_HIGH_EVAL_MODE':'slow','CUDA_VISIBLE_DEVICES':''}
        process=REAL_POPEN([sys.executable,'-B',str(SELF),'--fake-high','--run-dir',str(r['eval'])],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+30
            while not (d/'cpu-slow-worker.json').exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            self.assertTrue((d/'cpu-slow-worker.json').exists());process.send_signal(signal.SIGTERM);self.assertEqual(process.wait(timeout=40),130)
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=40)
            log.close()
        s=read_json(r['eval']/'eval-summary.json');self.assertEqual(s['status'],'cancelled');self.assertFalse(s['result_files']);receipt=read_json(d/'execution.json');self.assertTrue(receipt['cleanup_completed']);self.assertEqual(record_status(d)['status'],'cancelled')
        pids=[process.pid,read_json(d/'cpu-high-job.json')['pid'],read_json(d/'cpu-slow-worker.json')['pid']]
        self.assertTrue(all(not Path('/proc/'+str(pid)).exists() for pid in pids));write_json_atomic(self.case/'pid-audit.json',{'pids':pids,'all_exited':True})

def main():
    global RESULT
    p=argparse.ArgumentParser();p.add_argument('--result-root',required=True,type=Path);p.add_argument('--test',action='append',default=[]);a=p.parse_args();RESULT=a.result_root.resolve()
    if not RESULT.is_relative_to((hi.REPO/'smoke/results').resolve()):p.error('smoke results only')
    RESULT.mkdir(parents=True,exist_ok=False);hi.RESULT=RESULT;fx.RESULT=RESULT
    names=a.test or [n for n in sorted(HighTests.__dict__) if n.startswith('test_')];start=time.monotonic();res=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(HighTests(n) for n in names))
    write_json_atomic(RESULT/'summary.json',{'tests':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),'successful':res.wasSuccessful(),'elapsed_seconds':time.monotonic()-start,'details':[{'test':str(t),'traceback':s} for t,s in res.failures+res.errors]});return 0 if res.wasSuccessful() else 1
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fake-job':raise SystemExit(fake_job(sys.argv[2:]))
    if len(sys.argv)>1 and sys.argv[1]=='--slow-worker':
        request=Path(sys.argv[2]);write_json_atomic(request.parent/'cpu-slow-worker.json',{'pid':os.getpid()});time.sleep(30);raise SystemExit(fx.fake_worker(request))
    if len(sys.argv)>1 and sys.argv[1]=='--fake-high':
        with patch.object(orch.subprocess,'Popen',side_effect=dispatch):raise SystemExit(run_eval.main(sys.argv[2:]))
    raise SystemExit(main())
