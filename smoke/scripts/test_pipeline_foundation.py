"""CPU acceptance tests for pipeline configuration, plans, records and manifests.

All generated inputs and outputs remain below --result-dir. No real models or
COSINE data are opened. Production modules must not import model/GPU packages.
"""
import argparse
from copy import deepcopy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
BLOCKED_IMPORTS = {'torch', 'vllm', 'transformers', 'deepspeed', 'wandb', 'peft', 'qwen_vl_utils'}


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED_IMPORTS:
            raise ImportError('CPU foundation test forbids model import: ' + fullname)


sys.meta_path.insert(0, NoModelImports())
REPO = Path(__file__).resolve().parents[2]
LIBROOT = REPO / 'scripts/experiments'
sys.path.insert(0, str(LIBROOT))
import yaml
import pipeline
from lib.artifacts import check_manifest, make_manifest, path_identity
from lib.common import TASKS, fingerprint, read_json, require_id, write_json_atomic
from lib.config import (ConfigError, deep_merge, dotted_overrides, load_config,
                        resolve_job_config, resolve_run_config, validate_snapshot)
from lib.records import (bind_inputs, create_experiment, create_job, create_stage_run,
                         load_experiment, load_job, load_stage_run, record_status,
                         register_artifact, update_status)

RESULT = None
COUNTS = {
    'cl_mhpnvg': (6, 6, 36), 'cl_custom': (6, 6, 36),
    'zero_shot': (0, 1, 6), 'multitask': (1, 1, 6),
    'singletask': (6, 6, 6), 'transfer_round1': (36, 30, 30),
    'transfer_round2': (36, 30, 30),
}


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.case = RESULT / 'cases' / self._testMethodName
        self.case.mkdir(parents=True, exist_ok=False)
        self.sequence = 0

    def config(self, template='cl_mhpnvg', patch=None, cli=None):
        self.sequence += 1
        document = yaml.safe_load((LIBROOT / 'configs' / (template + '.yaml')).read_text())
        roots = {key: str(self.case / key) for key in (
            'dataset_root', 'base_model', 'processor_path', 'judge_model', 'records_root',
            'checkpoints_root', 'infer_root', 'eval_root')}
        document = deep_merge(document, {'experiment': roots})
        if patch:
            document = deep_merge(document, patch)
        path = self.case / ('setting-' + str(self.sequence) + '.yaml')
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding='utf-8')
        return load_config(path, cli_overrides=cli)

    def run_record(self, stage='tune', snapshot=None, plan=None):
        snapshot = snapshot or self.config()
        exp = create_experiment(snapshot, 'acceptance')
        run = create_stage_run(exp, stage, run_id='run-' + stage, stage_plan=plan)
        return exp, run

    def file(self, name, value='fake fixture\n'):
        path = self.case / 'payloads' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')
        return path

    def directory(self, name, kind='adapter'):
        path = self.case / 'payloads' / name
        path.mkdir(parents=True)
        if kind in ('adapter', 'checkpoint'):
            (path / 'adapter_config.json').write_text('{}')
            (path / 'adapter_model.safetensors').write_bytes(b'FAKE-CPU-ONLY')
        elif kind in ('base_model', 'judge_model'):
            (path / 'config.json').write_text('{}')
        else:
            (path / 'preprocessor_config.json').write_text('{}')
        if kind == 'checkpoint':
            (path / 'trainer_state.json').write_text('{"global_step": 1}')
        return path

    def external_inputs(self, job):
        record, _ = load_job(job)
        manifests = {}
        for spec in record['input_specs']:
            path = Path(spec['path'])
            self.assertTrue(path.is_relative_to(self.case))
            if spec['kind'] in ('base_model', 'judge_model', 'processor'):
                path.mkdir(parents=True, exist_ok=True)
                (path / 'config.json').write_text('{}')
                (path / 'preprocessor_config.json').write_text('{}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('[]')
            manifests[spec['role']] = make_manifest(spec['kind'], path,
                source=spec['source'], metadata=spec['metadata'])
        return manifests

    def produced(self, job, kind, name, metadata=None):
        record, _ = load_job(job)
        producer = {key: record[key] for key in ('experiment_id', 'stage', 'run_id', 'job_id')}
        path = self.directory(name, kind) if kind in ('adapter', 'checkpoint') else self.file(name)
        return make_manifest(kind, path, producer=producer, metadata=metadata)

    def test_default_setting_job_and_cli_precedence(self):
        snap = self.config(patch={'infer': {
            'params': {'batch_size': 2},
            'jobs': {'infer-motion': {'params': {'batch_size': 8}}}}})
        self.assertEqual(snap['config']['infer']['params']['batch_size'], 2)
        self.assertEqual(snap['config']['infer']['params']['max_new_tokens'], 256)
        self.assertEqual(resolve_job_config(snap, 'infer', 'infer-motion')['config']['infer']['params']['batch_size'], 8)
        self.assertEqual(resolve_job_config(snap, 'infer', 'infer-motion',
            job_override={'params': {'batch_size': 10}})['config']['infer']['params']['batch_size'], 10)
        snap = self.config(patch={'infer': {'jobs': {'infer-motion': {'params': {'batch_size': 8}}}}},
            cli={'infer': {'params': {'batch_size': 16},
                           'jobs': {'infer-motion': {'params': {'batch_size': 32}}}}})
        self.assertEqual(resolve_job_config(snap, 'infer', 'infer-motion')['config']['infer']['params']['batch_size'], 32)
        self.assertEqual(resolve_job_config(snap, 'infer', 'infer-hypo')['config']['infer']['params']['batch_size'], 16)
        final = resolve_job_config(snap, 'infer', 'infer-motion', cli_override={'params': {'batch_size': 64}})
        self.assertEqual(final['config']['infer']['params']['batch_size'], 64)
        self.assertEqual(snap['config']['infer']['params']['batch_size'], 16)

    def test_stage_cli_survives_job_resolution(self):
        snap = self.config(patch={'infer': {'jobs': {'infer-motion': {'params': {'batch_size': 8}}}}})
        run = resolve_run_config(snap, 'infer', {'params': {'batch_size': 3}})
        job = resolve_job_config(run, 'infer', 'infer-motion')
        self.assertEqual(job['config']['infer']['params']['batch_size'], 3)
        self.assertEqual(run['layers']['parent_snapshot'], snap['snapshot_fingerprint'])

    def test_false_zero_and_list_replacement(self):
        snap = self.config(patch={
            'tune': {'params': {'gradient_checkpointing': False, 'dataloader_num_workers': 0, 'weight_decay': 0}},
            'infer': {'params': {'do_sample': False, 'seed': 0},
                      'run': {'gpu_groups': [[2]], 'fail_fast': False}}})
        self.assertIs(snap['config']['tune']['params']['gradient_checkpointing'], False)
        self.assertEqual(snap['config']['tune']['params']['dataloader_num_workers'], 0)
        self.assertEqual(snap['config']['infer']['run']['gpu_groups'], [[2]])
        self.assertIs(snap['config']['infer']['run']['fail_fast'], False)
        before = {'a': {'x': 1, 'y': [1, 2]}, 'z': True}
        after = deep_merge(before, {'a': {'y': [3]}, 'z': False})
        self.assertEqual(after, {'a': {'x': 1, 'y': [3]}, 'z': False})
        after['a']['y'].append(4)
        self.assertEqual(before['a']['y'], [1, 2])

    def test_snapshot_fingerprints_and_source_hashes(self):
        snap = self.config()
        import hashlib
        for source in snap['sources']:
            self.assertEqual(source['sha256'], hashlib.sha256(Path(source['path']).read_bytes()).hexdigest())
        self.assertEqual(snap['config_fingerprint'], fingerprint(snap['config']))
        changed = deepcopy(snap)
        changed['config']['infer']['params']['batch_size'] = 7
        with self.assertRaisesRegex(ConfigError, 'fingerprint'):
            validate_snapshot(changed)
        changed = deepcopy(snap)
        changed['sources'][0]['sha256'] = 'bad'
        with self.assertRaisesRegex(ConfigError, 'fingerprint'):
            validate_snapshot(changed)

    def test_duplicate_yaml_and_required_setting_keys(self):
        path = self.case / 'bad.yaml'
        for text in ('experiment:\n  name: a\n  name: b\n  setting: cl\n',
                     'experiment:\n  name: a\n', 'experiment:\n  setting: cl\n',
                     'experiment:\n  name: a\n  setting: cl\ninfer:\n  params:\n    batch_size: null\n'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ConfigError):
                    load_config(path)

    def test_dotted_cli_and_conflicts(self):
        values = dotted_overrides(['infer.params.batch_size=2', 'tune.params.gradient_checkpointing=false',
                                    'infer.run.gpu_groups=[[2]]', 'eval.params.seed=0'])
        self.assertEqual(values['infer']['params']['batch_size'], 2)
        self.assertIs(values['tune']['params']['gradient_checkpointing'], False)
        self.assertEqual(dotted_overrides(['params.batch_size=2'], stage='infer'), {'params': {'batch_size': 2}})
        for items in (['infer.params.batch_size=2', 'infer.params.batch_size=4'],
                      ['infer.params={batch_size: 2}', 'infer.params.batch_size=4'],
                      ['infer.params.batch_size=2', 'infer.params={}'], ['infer..params=2'],
                      ['infer.params.batch_size'], ['infer.params.batch_size=null']):
            with self.subTest(items=items), self.assertRaises(ConfigError):
                dotted_overrides(items)

    def test_job_override_cannot_change_lineage(self):
        snap = self.config()
        for override in ({'experiment': {'base_model': '/tmp/another'}}, {'task': 'geo'}, {'jobs': {}}):
            with self.subTest(override=override), self.assertRaises(ConfigError):
                resolve_job_config(snap, 'tune', 'tune-01-motion', job_override=override)

    def test_unknown_job_and_invalid_override_fail_before_records(self):
        for patch in ({'tune': {'jobs': {'typo': {'params': {'learning_rate': 0.001}}}}},
                      {'infer': {'jobs': {'infer-motion': {'params': {'tensor_parallel_size': 2}}}}}):
            snap = self.config(patch=patch)
            with self.subTest(patch=patch), self.assertRaises(ConfigError):
                pipeline.prepare_records(snap, 'invalid-prepare')
            self.assertFalse(Path(snap['config']['experiment']['records_root']).exists())

    def test_disabled_stage_and_id_validation(self):
        snap = self.config('zero_shot')
        for operation in (lambda: pipeline.build_stage_plan(snap, 'tune'),
                          lambda: resolve_run_config(snap, 'tune'),
                          lambda: resolve_job_config(snap, 'tune', 'a')):
            with self.assertRaises(ConfigError):
                operation()
        for identifier in ('../escape', '.', '..', '', 'space id', '/absolute', 'a' * 97):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                require_id(identifier)

    def full_stage_config(self, template):
        if template == 'transfer_round2':
            # The shipped profile is source-bound infer/eval only; exercise its
            # full 36-job tuning topology with an explicit tune-enabled variant.
            with self.assertRaisesRegex(ConfigError, 'transfer_source_experiment_id'):
                self.config(template)
            return self.config(template, patch={'experiment': {'enabled_stages': ['tune', 'infer', 'eval']}})
        return self.config(template)

    def assert_plan(self, template):
        snap = self.full_stage_config(template)
        plan = pipeline.build_plan(snap)
        tune_count, infer_count, pairs_count = COUNTS[template]
        stages = plan['stages']
        self.assertEqual(len(stages.get('tune', {}).get('jobs', [])), tune_count)
        self.assertEqual(len(stages['infer']['jobs']), infer_count)
        self.assertEqual(len(stages['eval']['jobs']), 1)
        judge = stages['eval']['jobs'][0]
        self.assertEqual(len(judge['requires']), pairs_count)
        self.assertEqual(len(judge['lineage']['scoring_pairs']), pairs_count)
        self.assertEqual(judge['lineage']['judge_model_loads'], 1)
        infer = stages['infer']['jobs']
        expected = {(j['job_id'], j['output']['slot'], task) for j in infer for task in j['tasks']}
        self.assertEqual(len(expected), pairs_count)
        self.assertEqual({(p['prediction_job'], p['prediction_slot'], p['task'])
                          for p in judge['lineage']['scoring_pairs']}, expected)
        self.assertEqual({(p['job_id'], p['slot'], p['task']) for p in judge['requires']}, expected)
        tune = stages.get('tune', {}).get('jobs', [])
        seen = {}
        for job in tune:
            self.assertTrue(set(job['depends_on']).issubset(seen))
            for ref in job['requires']:
                self.assertEqual(seen[ref['job_id']]['output']['slot'], ref['slot'])
            self.assertEqual(job['lineage']['training_state'], 'fresh_optimizer_scheduler_epoch_global_step')
            annotation = next(s for s in job['input_specs'] if s['kind'] == 'annotation')
            self.assertEqual(annotation['metadata'], {'task': job['task'], 'split': 'train'})
            self.assertEqual(Path(annotation['path']), Path(snap['config']['experiment']['dataset_root']) / job['task'] / 'train.json')
            seen[job['job_id']] = job
        for job in infer:
            self.assertEqual(job['lineage']['model_loads_per_job'], 1)
            self.assertEqual(job['output']['tasks'], job['tasks'])
            self.assertEqual({s['metadata']['task'] for s in job['input_specs'] if s['kind'] == 'annotation'}, set(job['tasks']))
            for ref in job['requires']:
                self.assertEqual(seen[ref['job_id']]['output']['slot'], ref['slot'])
        if template.startswith('cl_'):
            self.assertEqual([j['task'] for j in tune], list(TASKS))
            for index, job in enumerate(tune):
                self.assertEqual(job['depends_on'], [] if index == 0 else [tune[index-1]['job_id']])
            self.assertTrue(all(j['tasks'] == list(TASKS) for j in infer))
        elif template == 'zero_shot':
            self.assertNotIn('tune', stages)
            self.assertEqual(infer[0]['requires'], [])
            self.assertIsNone(infer[0]['lineage']['adapter'])
        elif template == 'multitask':
            self.assertEqual(tune[0]['task'], 'whole')
            self.assertEqual(infer[0]['tasks'], list(TASKS))
        elif template == 'singletask':
            self.assertTrue(all(not j['depends_on'] and not j['requires'] for j in tune))
            self.assertEqual({tuple(j['tasks']) for j in infer}, {(t,) for t in TASKS})
        else:
            self.assertEqual(len([j for j in tune if j['lineage']['round'] == 1]), 6)
            self.assertEqual(len([j for j in tune if j['lineage']['round'] == 2]), 30)
            pairset = set()
            for job in infer:
                source, target = job['lineage']['source_task'], job['lineage']['second_round_task']
                self.assertNotEqual(source, target)
                pairset.add((source, target))
                parent = seen[job['requires'][0]['job_id']]
                self.assertEqual(parent['task'], target)
                self.assertEqual(parent['depends_on'], ['tune-round1-' + source])
                self.assertEqual(job['tasks'], [source if template == 'transfer_round1' else target])
            self.assertEqual(pairset, {(s, t) for s in TASKS for t in TASKS if s != t})

    def test_custom_cl_order(self):
        order = list(reversed(TASKS))
        snap = self.config('cl_custom', patch={'experiment': {'task_order': order}})
        jobs = pipeline.build_stage_plan(snap, 'tune')['jobs']
        self.assertEqual([j['task'] for j in jobs], order)
        for index, job in enumerate(jobs):
            self.assertEqual(job['depends_on'], [] if not index else [jobs[index-1]['job_id']])
            if index:
                self.assertEqual(job['requires'][0]['slot'], order[index-1])

    def test_prepare_all_templates_and_no_workloads(self):
        for template, counts in COUNTS.items():
            with self.subTest(template=template):
                snap = self.full_stage_config(template)
                exp = pipeline.prepare_records(snap, 'prepare-' + template)
                record, saved = load_experiment(exp)
                self.assertEqual(saved, snap)
                self.assertEqual(record['status'], 'active')
                self.assertEqual(len(record['stage_runs']), 2 if template == 'zero_shot' else 3)
                for link in record['stage_runs']:
                    run_dir = exp / link['path']
                    run, _ = load_stage_run(run_dir)
                    self.assertEqual(run['status'], 'planned')
                    self.assertFalse(Path(run['output_root']).exists())
                    for item in run['jobs']:
                        job, effective = load_job(run_dir / item['path'])
                        self.assertEqual(job['status'], 'planned')
                        self.assertEqual(job['inputs'], {})
                        self.assertEqual(job['artifacts'], [])
                        self.assertFalse(Path(job['output_dir']).exists())
                        validate_snapshot(effective)
                        if job['stage'] == 'tune':
                            self.assertIn('reset_optimizer' if job['requires'] else 'fresh_optimizer', job['training_state_policy'])
                with self.assertRaises(FileExistsError):
                    pipeline.prepare_records(snap, 'prepare-' + template)

    def test_real_cli_plan_prepare_and_invalid_argument(self):
        snap = self.config('zero_shot')
        setting = Path(snap['sources'][1]['path'])
        command = [sys.executable, '-B', str(LIBROOT/'pipeline.py')]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        outputs = []
        for args in (['plan', '--config', str(setting), '--set', 'infer.params.batch_size=2'],
                     ['prepare', '--config', str(setting), '--experiment-id', 'cli-prepare'],
                     ['prepare', '--config', str(setting), '--experiment-id', 'bad-cli', '--set', 'infer.run.gpu_groups=[[0]]']):
            process = subprocess.run(command + args, env=env, cwd=self.case, text=True, capture_output=True, timeout=30)
            outputs.append({'args': args, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        write_json_atomic(self.case / 'cli-results.json', outputs)
        self.assertEqual(outputs[0]['returncode'], 0, outputs[0]['stderr'])
        self.assertEqual(json.loads(outputs[0]['stdout'])['setting'], 'zero_shot')
        self.assertEqual(outputs[1]['returncode'], 0, outputs[1]['stderr'])
        self.assertEqual(load_experiment(Path(outputs[1]['stdout'].strip()))[0]['experiment_id'], 'cli-prepare')
        self.assertNotEqual(outputs[2]['returncode'], 0)
        self.assertIn('GPU 0 is forbidden', outputs[2]['stderr'])
        self.assertFalse((Path(snap['config']['experiment']['records_root']) / 'bad-cli').exists())

    def test_saved_snapshot_independent_of_yaml_changes(self):
        snap = self.config()
        exp = create_experiment(snap, 'frozen')
        Path(snap['sources'][1]['path']).write_text('broken: [yaml')
        run = create_stage_run(exp, 'tune')
        job = create_job(run, 'tune-motion', task='motion')
        self.assertEqual(load_job(job)[1]['config']['tune']['params']['learning_rate'], 0.0005)
        self.assertEqual(load_experiment(exp)[1], snap)

    def test_snapshot_and_plan_tampering_rejected_on_load(self):
        snap = self.config()
        exp, run = self.run_record(snapshot=snap, plan=pipeline.build_stage_plan(snap, 'tune'))
        plan_path = run / 'plan.json'
        plan = read_json(plan_path)
        plan['jobs'][0]['task'] = 'geo'
        write_json_atomic(plan_path, plan)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_stage_run(run)
        snapshot_path = exp / 'config.snapshot.json'
        frozen = read_json(snapshot_path)
        frozen['config']['infer']['params']['batch_size'] = 2
        write_json_atomic(snapshot_path, frozen)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_experiment(exp)

    def test_record_uniqueness_dependencies_and_terminal_guard(self):
        exp, run = self.run_record()
        with self.assertRaises(ValueError):
            create_stage_run(exp, 'tune')
        a = create_job(run, 'a', task='motion')
        for kwargs in ({'depends_on': ['missing']}, {'depends_on': ['bad']}, {'depends_on': ['a', 'a']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                create_job(run, 'bad', **kwargs)
        with self.assertRaises(ValueError):
            create_job(run, 'a')
        b = create_job(run, 'b', task='hypo', depends_on=['a'])
        with self.assertRaises(ValueError):
            update_status(b, 'running')
        update_status(a, 'running')
        update_status(a, 'succeeded')
        update_status(b, 'running')
        update_status(b, 'failed', 'expected simulated failure')
        with self.assertRaises(ValueError):
            update_status(b, 'running')
        with self.assertRaises(ValueError):
            update_status(exp, 'succeeded')
        update_status(run, 'running')
        with self.assertRaises(ValueError):
            update_status(run, 'succeeded')
        update_status(run, 'failed')
        with self.assertRaises(ValueError):
            create_job(run, 'late')
        update_status(exp, 'failed')
        with self.assertRaises(ValueError):
            create_stage_run(exp, 'infer')

    def test_full_cl_two_jobs_explicit_adapter_binding(self):
        snap = self.config()
        stage_plan = pipeline.build_stage_plan(snap, 'tune')
        # This isolated two-job lifecycle deliberately has no six-job stage plan.
        exp, run = self.run_record(snapshot=snap)
        jobs = [create_job(run, **plan) for plan in stage_plan['jobs'][:2]]
        first, second = jobs
        self.assertIn('base_model_fresh', load_job(first)[0]['training_state_policy'])
        self.assertIn('continue_existing_adapter', load_job(second)[0]['training_state_policy'])
        update_status(run, 'running')
        with self.assertRaisesRegex(ValueError, 'external input'):
            update_status(first, 'running')
        first_inputs = self.external_inputs(first)
        bind_inputs(first, first_inputs)
        update_status(first, 'running')
        with self.assertRaisesRegex(ValueError, 'primary output'):
            update_status(first, 'succeeded')
        parent = self.produced(first, 'adapter', 'motion-final', {'slot': 'motion', 'training_task': 'motion'})
        register_artifact(first, 'final-adapter', parent)
        second_inputs = self.external_inputs(second)
        with self.assertRaisesRegex(ValueError, 'not explicitly bound'):
            bind_inputs(second, second_inputs)
        foreign = deepcopy(parent)
        foreign['producer']['experiment_id'] = 'foreign'
        foreign['fingerprint'] = fingerprint({k:v for k,v in foreign.items() if k != 'fingerprint'})
        with self.assertRaisesRegex(ValueError, 'not explicitly bound'):
            bind_inputs(second, {**second_inputs, 'parent-adapter': foreign})
        bind_inputs(second, {**second_inputs, 'parent-adapter': parent})
        with self.assertRaisesRegex(ValueError, 'dependencies must succeed'):
            update_status(second, 'running')
        update_status(first, 'succeeded')
        update_status(second, 'running')
        child = self.produced(second, 'adapter', 'hypo-final', {'slot': 'hypo', 'training_task': 'hypo'})
        register_artifact(second, 'final-adapter', child)
        update_status(second, 'succeeded')
        update_status(run, 'succeeded')
        update_status(exp, 'succeeded')
        self.assertEqual(record_status(exp)['status'], 'succeeded')

    def test_stage_plan_rejects_unplanned_job(self):
        snap = self.config()
        plan = pipeline.build_stage_plan(snap, 'tune')
        _, run = self.run_record(snapshot=snap, plan=plan)
        with self.assertRaises(ValueError, msg='A frozen six-job plan must reject an unplanned job'):
            create_job(run, 'not-in-plan', task='motion')

    def test_stage_plan_rejects_changed_task(self):
        snap = self.config()
        plan = pipeline.build_stage_plan(snap, 'tune')
        _, run = self.run_record(snapshot=snap, plan=plan)
        changed = deepcopy(plan['jobs'][0])
        changed['task'] = 'geo'
        with self.assertRaises(ValueError, msg='A planned motion job must reject a geo task binding'):
            create_job(run, **changed)

    def test_stage_success_requires_all_planned_jobs(self):
        snap = self.config()
        plan = pipeline.build_stage_plan(snap, 'tune')
        _, run = self.run_record(snapshot=snap, plan=plan)
        first = create_job(run, **plan['jobs'][0])
        bind_inputs(first, self.external_inputs(first))
        update_status(run, 'running')
        update_status(first, 'running')
        adapter = self.produced(first, 'adapter', 'motion-final', {'slot': 'motion', 'training_task': 'motion'})
        register_artifact(first, 'final-adapter', adapter)
        update_status(first, 'succeeded')
        with self.assertRaises(ValueError, msg='A six-job stage must not succeed with only one completed job'):
            update_status(run, 'succeeded')

    def test_input_spec_mismatch_and_rebinding_rejected(self):
        snap = self.config()
        _, run = self.run_record(snapshot=snap)
        plan = pipeline.build_stage_plan(snap, 'tune')['jobs'][0]
        job = create_job(run, **plan)
        inputs = self.external_inputs(job)
        role = 'annotation-motion-train'
        bad = deepcopy(inputs)
        spec = next(s for s in plan['input_specs'] if s['role'] == role)
        bad[role] = make_manifest('annotation', spec['path'], source='wrong-source', metadata=spec['metadata'])
        with self.assertRaises(ValueError):
            bind_inputs(job, bad)
        bad[role] = make_manifest('annotation', spec['path'], source=spec['source'], metadata={'task': 'geo', 'split': 'train'})
        with self.assertRaises(ValueError):
            bind_inputs(job, bad)
        bind_inputs(job, inputs)
        with self.assertRaises(ValueError):
            bind_inputs(job, inputs)
        update_status(job, 'running')
        with self.assertRaises(ValueError):
            bind_inputs(job, inputs)

    def test_checkpoint_resume_same_task_config_and_explicit_only(self):
        _, run = self.run_record()
        job = create_job(run, 'resume-motion', task='motion')
        record, _ = load_job(job)
        checkpoint_path = self.directory('checkpoint-1', 'checkpoint')
        valid = {'training_task': 'motion', 'config_fingerprint': record['config_fingerprint']}
        for metadata in ({**valid, 'training_task': 'hypo'}, {**valid, 'config_fingerprint': 'wrong'}):
            ckpt = make_manifest('checkpoint', checkpoint_path, source='fixture', metadata=metadata)
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                bind_inputs(job, {'resume': ckpt}, resume_training_state=True)
        ckpt = make_manifest('checkpoint', checkpoint_path, source='fixture', metadata=valid)
        with self.assertRaises(ValueError):
            bind_inputs(job, {'resume': ckpt}, resume_training_state=False)
        with self.assertRaises(ValueError):
            bind_inputs(job, {}, resume_training_state=True)
        bound = bind_inputs(job, {'resume': ckpt}, resume_training_state=True)
        self.assertEqual(bound['training_state_policy'], 'restore_full_checkpoint')
        self.assertTrue(bound['resume_training_state'])
        other = create_job(run, 'fresh-motion', task='motion')
        self.assertEqual(load_job(other)[0]['inputs'], {})
        self.assertIn('base_model_fresh', load_job(other)[0]['training_state_policy'])
        adapter = make_manifest('adapter', self.directory('parent'), source='fixture')
        child = create_job(run, 'continue-hypo', task='hypo', inputs={'parent': adapter})
        self.assertIn('continue_existing_adapter_reset_optimizer_scheduler_epoch_global_step',
                      load_job(child)[0]['training_state_policy'])
        self.assertFalse(load_job(child)[0]['resume_training_state'])
        with self.assertRaises(ValueError):
            create_job(run, 'two-adapters', inputs={'a': adapter, 'b': adapter})
        with self.assertRaises(ValueError):
            create_job(run, 'not-bool', resume_training_state=1)

    def test_infer_requires_prediction_for_each_task(self):
        exp, run = self.run_record('infer')
        job = create_job(run, 'infer-two', tasks=['motion', 'hypo'],
                         output={'artifact_kind': 'prediction', 'slot': 'test', 'tasks': ['motion', 'hypo']})
        first = self.produced(job, 'prediction', 'motion.jsonl', {'slot': 'test', 'task': 'motion'})
        with self.assertRaises(ValueError):
            register_artifact(job, 'motion', first)
        update_status(run, 'running')
        update_status(job, 'running')
        register_artifact(job, 'motion', first)
        with self.assertRaises(ValueError):
            update_status(job, 'succeeded')
        with self.assertRaises(ValueError):
            register_artifact(job, 'duplicate-task', first)
        for metadata in ({'slot': 'wrong', 'task': 'hypo'}, {'slot': 'test', 'task': 'geo'}):
            manifest = self.produced(job, 'prediction', 'bad-' + metadata['task'], metadata)
            with self.assertRaises(ValueError):
                register_artifact(job, 'wrong', manifest)
        second = self.produced(job, 'prediction', 'hypo.jsonl', {'slot': 'test', 'task': 'hypo'})
        register_artifact(job, 'hypo', second)
        update_status(job, 'succeeded')
        update_status(run, 'succeeded')
        update_status(exp, 'succeeded')

    def test_eval_bind_all_predictions_and_metrics_lifecycle(self):
        snap = self.config('zero_shot')
        exp = pipeline.prepare_records(snap, 'zeroshot-fixture')
        runs = {r['stage']: exp / r['path'] for r in load_experiment(exp)[0]['stage_runs']}
        infer = runs['infer'] / 'jobs/infer-base'
        judge = runs['eval'] / 'jobs/eval-judge-all'
        bind_inputs(infer, self.external_inputs(infer))
        update_status(runs['infer'], 'running')
        update_status(infer, 'running')
        predictions = {}
        for task in TASKS:
            manifest = self.produced(infer, 'prediction', task + '.jsonl', {'slot': 'base', 'task': task})
            register_artifact(infer, task, manifest)
            predictions['prediction-' + task] = manifest
        update_status(infer, 'succeeded')
        update_status(runs['infer'], 'succeeded')
        inputs = self.external_inputs(judge)
        with self.assertRaises(ValueError):
            bind_inputs(judge, {**inputs, **{k:v for k,v in predictions.items() if k != 'prediction-geo'}})
        bind_inputs(judge, {**inputs, **predictions})
        update_status(runs['eval'], 'running')
        update_status(judge, 'running')
        with self.assertRaises(ValueError):
            update_status(judge, 'succeeded')
        for kind in ('judge_raw', 'judge_scored', 'metrics'):
            manifest = self.produced(judge, kind, kind + '.json', {'slot': 'combined'})
            register_artifact(judge, kind, manifest)
        update_status(judge, 'succeeded')
        update_status(runs['eval'], 'succeeded')
        update_status(exp, 'succeeded')
        self.assertEqual(record_status(exp)['status'], 'succeeded')

    def test_registration_wrong_producer_kind_and_stale_payload(self):
        _, run = self.run_record('infer')
        job = create_job(run, 'infer', tasks=['motion'],
                         output={'artifact_kind': 'prediction', 'slot': 'test', 'tasks': ['motion']})
        update_status(job, 'running')
        manifest = self.produced(job, 'prediction', 'valid.jsonl', {'slot': 'test', 'task': 'motion'})
        wrong = deepcopy(manifest)
        wrong['producer']['run_id'] = 'another-run'
        wrong['fingerprint'] = fingerprint({k:v for k,v in wrong.items() if k != 'fingerprint'})
        with self.assertRaises(ValueError):
            register_artifact(job, 'wrong-producer', wrong)
        adapter = self.produced(job, 'adapter', 'wrong-kind')
        with self.assertRaises(ValueError):
            register_artifact(job, 'wrong-kind', adapter)
        Path(manifest['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'contents no longer match'):
            register_artifact(job, 'stale', manifest)
        self.assertEqual(load_job(job)[0]['artifacts'], [])

    def test_manifests_all_kinds_full_hashes_and_symlink(self):
        for kind in ('base_model', 'judge_model', 'processor', 'adapter', 'checkpoint'):
            path = self.directory(kind, kind)
            manifest = make_manifest(kind, path, source='synthetic CPU fixture')
            self.assertEqual(check_manifest(manifest, verify_contents=True), manifest)
            self.assertEqual(manifest['identity']['hash_method'], 'sha256-all-files')
        for kind in ('annotation', 'prediction', 'judge_raw', 'judge_scored', 'metrics', 'summary'):
            path = self.file(kind + '.json')
            manifest = make_manifest(kind, path, source='synthetic CPU fixture')
            self.assertEqual(check_manifest(manifest, verify_contents=True), manifest)
        target = self.file('target.json')
        link = self.case / 'linked.json'
        link.symlink_to(target)
        manifest = make_manifest('annotation', link, source='fixture')
        self.assertEqual(manifest['requested_path'], str(link))
        self.assertEqual(manifest['path'], str(target.resolve()))
        directory = self.directory('with-file-symlink', 'processor')
        (directory/'linked.json').symlink_to(target)
        entries = path_identity(directory)['files']
        self.assertEqual(next(e for e in entries if e['name'] == 'linked.json')['symlink_target'], str(target))

    def test_manifest_tamper_and_content_changes(self):
        path = self.directory('adapter')
        manifest = make_manifest('adapter', path, source='fixture')
        tampered = deepcopy(manifest)
        tampered['metadata']['unexpected'] = True
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            check_manifest(tampered)
        (path/'adapter_model.safetensors').write_bytes(b'DIFFERENT')
        with self.assertRaisesRegex(ValueError, 'contents no longer match'):
            check_manifest(manifest, verify_contents=True)
        path = self.file('annotation.json', '[]')
        manifest = make_manifest('annotation', path, source='fixture')
        path.write_text('[1]')
        with self.assertRaisesRegex(ValueError, 'contents no longer match'):
            check_manifest(manifest, verify_contents=True)

    def test_manifest_shape_provenance_and_invalid_directories(self):
        path = self.directory('not-checkpoint')
        for kwargs in ({}, {'producer': {}, 'source': 'both'}, {'source': ''}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_manifest('adapter', path, **kwargs)
        with self.assertRaises(ValueError):
            make_manifest('checkpoint', path, source='fixture')
        empty = self.case/'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            make_manifest('processor', empty, source='fixture')
        with self.assertRaises(ValueError):
            make_manifest('base_model', path, source='fixture')
        with self.assertRaises(ValueError):
            make_manifest('adapter', self.file('file'), source='fixture')
        (path/'directory-link').symlink_to(empty, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Directory symlinks'):
            path_identity(path)
        (path/'directory-link').unlink()
        (path/'broken').symlink_to(self.case/'missing')
        with self.assertRaisesRegex(ValueError, 'broken symlink'):
            path_identity(path)

    def test_strict_json_and_atomic_write(self):
        path = self.case/'record.json'
        value = {'中文': [0, False, 'motion'], 'nested': {'count': 1}}
        write_json_atomic(path, value)
        self.assertEqual(read_json(path), value)
        self.assertEqual(list(self.case.glob('.record.json-*')), [])
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_json(path)


INVALID_CONFIGS = {
    'unknown_root': {'unknown': {}},
    'unknown_param': {'infer': {'params': {'batc_size': 2}}},
    'null_value': {'tune': {'params': {'learning_rate': None}}},
    'numeric_string': {'tune': {'params': {'learning_rate': '0.0005'}}},
    'boolean_as_integer': {'infer': {'params': {'batch_size': True}}},
    'nonfinite': {'tune': {'params': {'learning_rate': float('inf')}}},
    'nan': {'tune': {'params': {'learning_rate': float('nan')}}},
    'negative': {'tune': {'params': {'weight_decay': -1}}},
    'zero_batch': {'infer': {'params': {'batch_size': 0}}},
    'fractional_batch': {'infer': {'params': {'batch_size': 1.5}}},
    'invalid_ratio': {'infer': {'params': {'gpu_memory_utilization': 1.1}}},
    'dropout_one': {'tune': {'params': {'lora_dropout': 1}}},
    'relative_path': {'experiment': {'base_model': 'relative/model'}},
    'parent_path': {'experiment': {'infer_root': '/tmp/../escape'}},
    'unknown_setting': {'experiment': {'setting': 'typo'}},
    'missing_task': {'experiment': {'tasks': ['motion']}},
    'duplicate_task': {'experiment': {'task_order': ['motion'] * 6}},
    'unknown_task': {'experiment': {'task_order': ['motoin', *TASKS[1:]]}},
    'empty_stages': {'experiment': {'enabled_stages': []}},
    'unknown_stage': {'experiment': {'enabled_stages': ['tune', 'predict']}},
    'gpu0_tune': {'tune': {'run': {'gpu_groups': [[0]]}}},
    'gpu0_infer': {'infer': {'run': {'gpu_groups': [[0]]}}},
    'gpu0_eval': {'eval': {'run': {'gpu_groups': [[0]]}}},
    'gpu_out_of_range': {'tune': {'run': {'gpu_groups': [[4]]}}},
    'gpu_duplicate': {'tune': {'run': {'gpu_groups': [[2, 2]]}}},
    'gpu_overlapping_groups': {'tune': {'run': {'gpu_groups': [[2], [2, 3]]}}},
    'empty_gpu_group': {'tune': {'run': {'gpu_groups': [[]]}}},
    'parallel_exceeds_groups': {'tune': {'run': {'max_parallel_jobs': 2}}},
    'infer_tp_mismatch': {'infer': {'params': {'tensor_parallel_size': 2}}},
    'eval_tp_mismatch': {'eval': {'params': {'tensor_parallel_size': 2}}},
    'infer_context_too_short': {'infer': {'params': {'max_model_len': 256}}},
    'eval_context_too_short': {'eval': {'params': {'max_model_len': 128}}},
    'image_range': {'tune': {'params': {'min_pixels': 50177}}},
    'frame_range': {'infer': {'params': {'video_min_frames': 9}}},
    'video_pixel_range': {'infer': {'params': {'video_min_pixels': 802817}}},
    'mixed_precision': {'tune': {'params': {'fp16': True}}},
    'lora_disabled': {'tune': {'params': {'lora_enable': False}}},
    'vision_unfrozen': {'tune': {'params': {'tune_mm_vision': True}}},
    'flatten_packing': {'tune': {'params': {'data_packing': True}}},
    'master_port': {'tune': {'run': {'master_port': 65536}}},
    'job_lineage_override': {'tune': {'jobs': {'tune-01-motion': {'task': 'geo'}}}},
    'zero_shot_tune': {'experiment': {'setting': 'zero_shot'}},
}


def make_invalid_test(patch):
    def test(self):
        with self.assertRaises(ConfigError):
            self.config(patch=patch)
    return test


def make_plan_test(template):
    def test(self):
        self.assert_plan(template)
    return test


for _name, _patch in INVALID_CONFIGS.items():
    setattr(FoundationTests, 'test_reject_' + _name, make_invalid_test(_patch))
for _template in COUNTS:
    setattr(FoundationTests, 'test_plan_' + _template, make_plan_test(_template))


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.events.append({'test': test.id(), 'status': 'passed'})
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.events.append({'test': test.id(), 'status': 'failed', 'traceback': self._exc_info_to_string(err, test)})
    def addError(self, test, err):
        super().addError(test, err)
        self.events.append({'test': test.id(), 'status': 'error', 'traceback': self._exc_info_to_string(err, test)})
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.events.append({'test': subtest.id(), 'status': 'subtest_failure', 'traceback': self._exc_info_to_string(err, test)})


def main():
    global RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True, type=Path)
    args = parser.parse_args()
    RESULT = args.result_dir.resolve(strict=True)
    if not RESULT.is_relative_to((REPO/'smoke/results').resolve(strict=True)):
        parser.error('--result-dir must remain below smoke/results')
    if (RESULT/'tests.json').exists() or (RESULT/'cases').exists():
        parser.error('Use a fresh result directory; prior test evidence is never overwritten')
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FoundationTests)
    with (RESULT/'tests.log').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=JsonResult).run(suite)
    forbidden = sorted(set(sys.modules).intersection(BLOCKED_IMPORTS))
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'successful': result.wasSuccessful() and not forbidden, 'elapsed_seconds': time.time()-started,
               'python': sys.version, 'executable': sys.executable, 'yaml': yaml.__version__,
               'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'forbidden_imports': forbidden,
               'events': result.events}
    write_json_atomic(RESULT/'tests.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'events'}, ensure_ascii=False, indent=2))
    if not summary['successful']:
        print((RESULT/'tests.log').read_text()[-24000:])
    return 0 if summary['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
