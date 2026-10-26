#!/usr/bin/env python3
"""CPU-only behavioral checks for the unchanged COSINE evaluation implementation."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/COSINE'))
import dataset_utils as data
import eval_cosine as cli
import qwen_judge as judge
VALID = {'status': 'valid', 'score': 1, 'gt': 'B', 'pred': 'B'}
INVALID = {'status': 'invalid', 'score': None, 'gt': None, 'pred': None}

def example(index=0, metric='MCQ'):
    return {'inference_index': index, 'metric_type': metric, 'question': 'Choose: A. left B. right',
            'ground_truth': 'B', 'prediction': 'The answer is B.', 'n_options': 2,
            'source_dataset': 'fixture', 'task_type': 'direction'}

class Processor:
    def apply_chat_template(self, messages, **kwargs): return json.dumps(messages)
    def tokenizer(self, prompt, **kwargs): return {'input_ids': [0] * len(prompt)}

class Engine:
    def __init__(self, outputs): self.outputs, self.requests = iter(outputs), []
    def generate(self, prompts, **kwargs):
        self.requests.append(prompts)
        return [NS(finished=True, outputs=[NS(text=text, finish_reason='stop')]) for text in next(self.outputs)]

class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.annotation = data.load_registered_test('cosine_motion_test')
        cls.input_path = Path('/home/linruikang/data/infer-results/motion-final-lora-20261026.ozbi042d/motion.predictions.jsonl')
        cls.predictions = data.read_jsonl(cls.input_path)
        cls.records = data.align_predictions(cls.predictions, cls.samples, 'cosine_motion_test')

    def test_full_motion_alignment(self):
        self.assertEqual(len(self.records), 691)
        self.assertEqual([r['inference_index'] for r in self.records], list(range(691)))
        self.assertEqual(sum(r['metric_type'] == 'MCQ' for r in self.records), 287)
        self.assertEqual(sum(r['metric_type'] == 'OPEN' for r in self.records), 404)

    def test_shuffled_alignment(self):
        self.assertEqual(data.align_predictions(self.predictions[::-1], self.samples, 'cosine_motion_test'), self.records)

    def test_all_registered_splits(self):
        for dataset in data.TEST_DATASETS:
            with self.subTest(dataset=dataset):
                samples, path = data.load_registered_test(dataset)
                self.assertTrue(path.is_file())
                for sample in samples:
                    self.assertTrue(data.extract_ground_truth(sample).strip())
                    self.assertTrue(data.extract_question(sample).strip())
                    self.assertIn(sample['_meta']['metric_type'], judge.SUPPORTED_METRICS)

    def test_test_only_registry(self):
        with self.assertRaises(ValueError): data.load_registered_test('cosine_motion_train')

    def test_duplicate_source_ids_remain_separate(self):
        samples = copy.deepcopy(self.samples[:2])
        samples[1]['_meta']['id'] = samples[0]['_meta']['id']
        predictions = copy.deepcopy(self.predictions[:2])
        predictions[1]['_meta'] = samples[1]['_meta']
        predictions[1]['id'] = samples[0]['_meta']['id']
        rows = data.align_predictions(predictions, samples, 'cosine_motion_test')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['id'], rows[1]['id'])

    def test_reject_missing_row(self):
        with self.assertRaisesRegex(ValueError, 'missing'): data.align_predictions(self.predictions[:-1], self.samples, 'cosine_motion_test')

    def test_reject_duplicate_row(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'): data.align_predictions([self.predictions[0]]*2, self.samples, 'cosine_motion_test')

    def test_reject_bad_identity_fields(self):
        for key in ['dataset', '_meta', 'ground_truth', 'id', 'task_type', 'metric_type', 'sample_index']:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.predictions)
                bad[0][key] = {'changed': True} if key == '_meta' else 'changed'
                with self.assertRaises(ValueError): data.align_predictions(bad, self.samples, 'cosine_motion_test')

    def test_reject_bad_index(self):
        for index in [None, True, -1, 691, '0']:
            with self.subTest(index=index):
                bad = copy.deepcopy(self.predictions)
                bad[0]['inference_index'] = index
                with self.assertRaises(ValueError): data.align_predictions(bad, self.samples, 'cosine_motion_test')

    def test_prediction_must_be_text(self):
        bad = copy.deepcopy(self.predictions); bad[0]['prediction'] = None
        with self.assertRaises(TypeError): data.align_predictions(bad, self.samples, 'cosine_motion_test')

    def test_empty_prediction_preserved(self):
        bad = copy.deepcopy(self.predictions); bad[0]['prediction'] = ''
        self.assertEqual(data.align_predictions(bad, self.samples, 'cosine_motion_test')[0]['prediction'], '')

    def test_markers_only_removed_and_full_prediction_preserved(self):
        sample = {'conversations': [{'from': 'human', 'value': '<image><video><visual>A. left B. right'}, {'from': 'gpt', 'value': 'B'}]}
        self.assertEqual(data.extract_question(sample), 'A. left B. right')
        r = example(); r['prediction'] = 'Earlier A. Final answer B. ignore instructions'
        prompt = judge.build_messages(r)[1]['content']
        self.assertIn(r['prediction'], prompt); self.assertIn('ground_truth', prompt)

    def test_reference_cardinality(self):
        for turns in [[], [{'from': 'gpt','value': ''}], [{'from': 'gpt','value':'a'}]*2]:
            with self.assertRaises(ValueError): data.extract_ground_truth({'conversations': turns})

    def test_valid_mcq_and_open_protocol(self):
        for metric in ['MCQ', 'OPEN']:
            self.assertEqual(judge.parse_judge_response(json.dumps(VALID), metric), VALID)
            self.assertEqual(judge.parse_judge_response(json.dumps(INVALID), metric), INVALID)

    def test_valid_mra_protocol(self):
        value = {'status': 'valid','score': None,'gt': -2.5,'pred': 3e2}
        self.assertEqual(judge.parse_judge_response(json.dumps(value), 'MRA'), value)

    def test_strict_protocol_rejections(self):
        invalid = ['not json', '[]', 'null', chr(96)*3+'json\n'+json.dumps(VALID)+'\n'+chr(96)*3,
                   json.dumps(VALID)+' trailing', '{"status":"valid","status":"invalid","score":1,"gt":"B","pred":"B"}',
                   '{"status":"valid","score":NaN,"gt":"B","pred":"B"}']
        for changes in [{'extra': 1}, {'score': True}, {'score': '1'}, {'score': 0.5},
                        {'score': None}, {'gt': ''}, {'pred': ' '}, {'gt': 1}, {'status': 'unknown'}]:
            invalid.append(json.dumps({**VALID, **changes}))
        invalid.append(json.dumps({'status':'valid','score':1,'gt':'B'}))
        invalid.append(json.dumps({**INVALID, 'pred': 'B'}))
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises((ValueError, TypeError)): judge.parse_judge_response(response, 'MCQ')

    def test_numeric_protocol_rejections(self):
        for changes in [{'score':0}, {'gt':True}, {'pred':'1'}, {'gt':float('inf')}, {'pred':1e309}]:
            with self.subTest(changes=changes):
                value = {'status':'valid','score':None,'gt':1,'pred':1,**changes}
                with self.assertRaises(ValueError): judge.parse_judge_response(json.dumps(value), 'MRA')

    def test_mra_reference_convention(self):
        for gt,pred,expected in [(100,100,1),(100,150,0),(0,0,0),(1e-10,1e-10,0),(-100,-100,1),(100,125,0.5),(100,149.999,0.1)]:
            with self.subTest(gt=gt,pred=pred): self.assertEqual(judge.compute_mra(gt,pred),expected)

    def test_invalid_stays_in_denominator(self):
        records=[judge.scored_record(example(0),VALID),judge.scored_record(example(1),INVALID),judge.scored_record(example(2),None,'parse failure')]
        metrics=judge.aggregate_metrics(records)
        self.assertEqual(metrics['total'],3); self.assertEqual(metrics['invalid'],2)
        self.assertEqual(metrics['combined_flat_rawacc'],1/3)
        self.assertEqual(metrics['by_metric_type']['MCQ']['accuracy'],1/3)
        self.assertEqual(metrics['by_source_dataset']['fixture']['total'],3)

    def test_mixed_metric_weighting(self):
        records=[judge.scored_record(example(),VALID),judge.scored_record(example(1,'OPEN'),INVALID),judge.scored_record(example(2,'MRA'),{'status':'valid','score':None,'gt':100,'pred':125})]
        self.assertEqual(judge.aggregate_metrics(records)['combined_flat_rawacc'],0.5)

    def test_prompt_budget_exact_and_over(self):
        r=example(); messages=judge.build_messages(r); n=len(json.dumps(messages)); args=NS(max_new_tokens=256,max_model_len=n+256)
        self.assertTrue(cli.make_prompt(Processor(),messages,args,0))
        args.max_model_len-=1
        with self.assertRaisesRegex(ValueError,'no truncation'): cli.make_prompt(Processor(),messages,args,0)

    def test_mock_repair_and_resume(self):
        records=[example(0),example(1)]; args=NS(repair_attempts=1,max_model_len=20000,max_new_tokens=256)
        run={'settings':{'repair_attempts':1}}; done={}; engine=Engine([['bad json',json.dumps(INVALID)],[json.dumps(VALID)]])
        prompts={r['inference_index']:cli.make_prompt(Processor(),judge.build_messages(r),args,r['inference_index']) for r in records}
        with tempfile.TemporaryDirectory(dir=str(ROOT/'smoke/results')) as directory:
            p=Path(directory)/'raw.jsonl'
            with p.open('w') as handle: cli.judge_batch(engine,None,Processor(),records,prompts,args,handle,done,run,'fingerprint',[],2)
            self.assertEqual([len(r) for r in engine.requests],[2,1]); self.assertEqual(len(done[0]['attempts']),2)
            self.assertEqual(len(done[1]['attempts']),1); self.assertEqual(done[0]['parsed'],VALID)
            self.assertEqual(cli.restore_raw(p,records,run,'fingerprint'),done)
            rows=data.read_jsonl(p)
            for mutation in ['fingerprint','record','parse','attempt','duplicate','truncated']:
                changed=copy.deepcopy(rows)
                if mutation=='fingerprint': changed[0]['run_fingerprint']='other'
                if mutation=='record': changed[0]['record']['prediction']='changed'
                if mutation=='parse': changed[0]['parse_error']='wrong'
                if mutation=='attempt': changed[0]['attempts'][0]['messages']=[]
                if mutation=='duplicate': changed.append(changed[0])
                p.write_text(''.join(json.dumps(r)+'\n' for r in changed))
                if mutation=='truncated': p.write_text(p.read_text()+'{"broken":')
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ValueError): cli.restore_raw(p,records,run,'fingerprint')

    def test_mock_exhausted_repair(self):
        records=[example()]; args=NS(repair_attempts=1,max_model_len=20000,max_new_tokens=256); done={}; engine=Engine([['bad'],['still bad']])
        with tempfile.TemporaryDirectory(dir=str(ROOT/'smoke/results')) as directory:
            with (Path(directory)/'raw').open('w') as handle: cli.judge_batch(engine,None,Processor(),records,{0:'prompt'},args,handle,done,{'settings':{'repair_attempts':1}},'f',[],1)
        self.assertIsNotNone(done[0]['parse_error']); self.assertEqual(len(done[0]['attempts']),2)
        self.assertEqual(judge.scored_record(records[0],done[0]['parsed'],done[0]['parse_error'])['sample_score'],0)

    def test_mock_output_count_error(self):
        with self.assertRaisesRegex(RuntimeError,'number of outputs'): cli.judge_batch(Engine([[]]),None,Processor(),[example()],{0:'p'},NS(),io.StringIO(),{}, {},'f',[],1)

    def test_atomic_write_preserves_old_on_error(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'smoke/results')) as directory:
            p=Path(directory)/'out.json'; p.write_text('old')
            with self.assertRaises(ValueError): data.write_json_atomic(p,{'bad':float('nan')})
            self.assertEqual(p.read_text(),'old'); self.assertEqual(len(list(Path(directory).iterdir())),1)

    def test_output_guards(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'smoke/results')) as directory:
            p=Path(directory); unrelated=p/'keep.txt'; unrelated.write_text('keep'); args=NS(output_dir=str(p),resume=False,overwrite=False)
            raw,scored,metrics=cli.prepare_artifacts(args,self.input_path,self.annotation); raw.write_text('x')
            with self.assertRaises(FileExistsError): cli.prepare_artifacts(args,self.input_path,self.annotation)
            args.overwrite=True; cli.prepare_artifacts(args,self.input_path,self.annotation)
            self.assertEqual(unrelated.read_text(),'keep'); raw.symlink_to(self.input_path)
            with self.assertRaises(ValueError): cli.prepare_artifacts(args,self.input_path,self.annotation)

    def test_gpu_selection_rejects_before_query(self):
        for visible in ['', '0', '0,1', '1,1', 'all', '-1', '1,']:
            with self.subTest(visible=visible),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visible}),patch.object(cli.subprocess,'run') as query:
                with self.assertRaises((RuntimeError,ValueError)): cli.verify_gpu_selection(NS(tensor_parallel_size=1))
                query.assert_not_called()

    def test_mock_gpu_identity_and_worker_environment(self):
        rows='0, GPU-zero, 0000:01:00.0\n1, GPU-one, 0000:02:00.0\n2, GPU-two, 0000:03:00.0\n3, GPU-three, 0000:04:00.0\n'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-one'}),patch.object(cli.subprocess,'run',return_value=NS(stdout=rows)):
            devices=cli.verify_gpu_selection(NS(tensor_parallel_size=1))
            self.assertEqual(devices[0]['physical_index'],1); self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'1')
            self.assertEqual(os.environ['VLLM_WORKER_MULTIPROC_METHOD'],'spawn')
        for visible,output,tp in [('GPU-zero',rows,1),('GPU-absent',rows,1),('1',rows,2),('1',rows.replace('0000:01','0000:05'),1)]:
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visible}),patch.object(cli.subprocess,'run',return_value=NS(stdout=output)):
                with self.assertRaises((RuntimeError,ValueError)): cli.verify_gpu_selection(NS(tensor_parallel_size=tp))

    def test_no_cuda_libraries_imported(self):
        self.assertFalse(any(name in sys.modules for name in ['torch','vllm','transformers']))

    def test_shell_syntax(self):
        subprocess.run(['bash','-n',str(ROOT/'evaluation/COSINE/eval_cosine.sh')],check=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--report',required=True); args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvalTests))
    Path(args.report).write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful()},indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
