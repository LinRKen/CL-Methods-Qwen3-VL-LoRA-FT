"""CPU regression tests for frozen pipeline plans; use synthetic records only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import unittest

import test_pipeline_foundation as foundation

from lib.common import fingerprint, read_json, write_json_atomic
from lib.records import (bind_inputs, create_experiment, create_job, create_stage_run,
                         load_job, load_stage_run, record_status, register_artifact,
                         update_status)


class PlanConstraintTests(foundation.FoundationTests):
    def small_run(self, jobs=None, plan_patch=None):
        snap = self.config()
        plan = {'stage': 'tune', 'config_fingerprint': snap['config_fingerprint'],
                'jobs': jobs if jobs is not None else [
                    {'job_id': 'a', 'task': 'motion'}, {'job_id': 'b', 'task': 'geo'}]}
        if plan_patch:
            plan.update(plan_patch)
        exp = create_experiment(snap, 'small-' + str(self.sequence))
        run = create_stage_run(exp, 'tune', run_id='run-tune', stage_plan=plan)
        return exp, run, plan

    def completed_small_run(self):
        exp, run, plan = self.small_run()
        update_status(run, 'running')
        for item in plan['jobs']:
            job = create_job(run, **item)
            update_status(job, 'running')
            update_status(job, 'succeeded')
        return exp, run, plan

    def assert_create_rejected_without_writes(self, run, kwargs, message):
        before = (run / 'run.json').read_bytes()
        plan_before = (run / 'plan.json').read_bytes()
        children = sorted(p.name for p in (run / 'jobs').iterdir())
        with self.assertRaisesRegex(ValueError, message):
            create_job(run, **kwargs)
        self.assertEqual((run / 'run.json').read_bytes(), before)
        self.assertEqual((run / 'plan.json').read_bytes(), plan_before)
        self.assertEqual(sorted(p.name for p in (run / 'jobs').iterdir()), children)

    def assert_success_rejected_without_writes(self, run, message):
        before = (run / 'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError, message):
            update_status(run, 'succeeded')
        self.assertEqual((run / 'run.json').read_bytes(), before)
        self.assertEqual(record_status(run)['status'], 'running')

    def test_frozen_unplanned_registration_has_no_side_effects(self):
        _, run, _ = self.small_run()
        self.assert_create_rejected_without_writes(run,
            {'job_id': 'outside', 'task': 'motion'}, 'not in the frozen')
        self.assertFalse((run / 'jobs/outside').exists())

    def test_frozen_duplicate_registration_has_no_side_effects(self):
        _, run, _ = self.small_run()
        job = create_job(run, 'a', task='motion')
        original = (job / 'job.json').read_bytes()
        self.assert_create_rejected_without_writes(run,
            {'job_id': 'a', 'task': 'motion'}, 'already registered')
        self.assertEqual((job / 'job.json').read_bytes(), original)

    def test_frozen_missing_jobs_names_and_status(self):
        _, run, plan = self.small_run()
        update_status(run, 'running')
        self.assert_success_rejected_without_writes(run, 'missing: a, b; unexpected: none')
        create_job(run, **plan['jobs'][0])
        self.assert_success_rejected_without_writes(run, 'missing: b; unexpected: none')

    def test_frozen_extra_registration_rejected_at_completion(self):
        _, run, _ = self.completed_small_run()
        record = read_json(run / 'run.json')
        record['jobs'].append({'job_id': 'outside', 'path': 'jobs/a'})
        write_json_atomic(run / 'run.json', record)
        self.assert_success_rejected_without_writes(run, 'missing: none; unexpected: outside')

    def test_frozen_duplicate_registration_rejected_at_completion(self):
        _, run, _ = self.completed_small_run()
        record = read_json(run / 'run.json')
        record['jobs'].append(deepcopy(record['jobs'][0]))
        write_json_atomic(run / 'run.json', record)
        self.assert_success_rejected_without_writes(run, 'duplicate registered job IDs')

    def test_frozen_registered_path_must_match_job_identity(self):
        _, run, _ = self.completed_small_run()
        record = read_json(run / 'run.json')
        record['jobs'][0]['path'] = record['jobs'][1]['path']
        write_json_atomic(run / 'run.json', record)
        self.assert_success_rejected_without_writes(run, 'does not match its job ID: a')

    def test_frozen_omitted_fields_accept_api_defaults(self):
        _, run, _ = self.small_run(jobs=[{'job_id': 'minimal'}])
        job = create_job(run, 'minimal', task=None, tasks=None,
                         depends_on=[], requires=[], input_specs=[], lineage={}, output=None)
        record, _ = load_job(job)
        for field in ('depends_on', 'requires', 'input_specs'):
            self.assertEqual(record[field], [])
        self.assertEqual(record['lineage'], {})
        for field in ('task', 'tasks', 'output_spec'):
            self.assertIsNone(record[field])
        update_status(run, 'running')
        update_status(job, 'running')
        update_status(job, 'succeeded')
        update_status(run, 'succeeded')

    def test_frozen_config_overrides_and_runtime_inputs_remain_supported(self):
        _, run, _ = self.small_run(jobs=[{'job_id': 'a', 'task': 'motion'}])
        original_plan = (run / 'plan.json').read_bytes()
        job = create_job(run, 'a', task='motion',
                         job_override={'params': {'learning_rate': 0.0007}},
                         cli_override={'params': {'learning_rate': 0.0008}})
        record, snapshot = load_job(job)
        self.assertEqual(snapshot['config']['tune']['params']['learning_rate'], 0.0008)
        parent = foundation.make_manifest('adapter', self.directory('external-parent'), source='synthetic')
        bound = bind_inputs(job, {'parent-adapter': parent})
        self.assertEqual(bound['training_state_policy'],
                         'continue_existing_adapter_reset_optimizer_scheduler_epoch_global_step')
        self.assertFalse(bound['resume_training_state'])
        self.assertEqual((run / 'plan.json').read_bytes(), original_plan)
        self.assertEqual(load_stage_run(run)[0]['config_fingerprint'],
                         read_json(run / 'plan.json')['config_fingerprint'])

    def test_frozen_full_cl_plan_completes_with_explicit_artifacts(self):
        snap = self.config()
        plan = foundation.pipeline.build_stage_plan(snap, 'tune')
        exp, run = self.run_record(snapshot=snap, plan=plan)
        update_status(run, 'running')
        adapters = {}
        for item in plan['jobs']:
            job = create_job(run, **item)
            inputs = self.external_inputs(job)
            for ref in item['requires']:
                inputs['parent-adapter'] = adapters[ref['job_id']]
            bound = bind_inputs(job, inputs)
            self.assertFalse(bound['resume_training_state'])
            update_status(job, 'running')
            output = self.produced(job, 'adapter', item['job_id'] + '-final',
                                   {'slot': item['output']['slot'], 'training_task': item['task']})
            register_artifact(job, 'final-adapter', output)
            update_status(job, 'succeeded')
            adapters[item['job_id']] = output
        update_status(run, 'succeeded')
        update_status(exp, 'succeeded')
        self.assertEqual(record_status(run)['status'], 'succeeded')
        self.assertEqual(len(record_status(run)['jobs']), 6)
        self.assertEqual(record_status(exp)['status'], 'succeeded')

    def test_frozen_tampering_rejected_at_all_record_boundaries(self):
        _, run, _ = self.completed_small_run()
        plan = read_json(run / 'plan.json')
        plan['jobs'][0]['task'] = 'hypo'
        write_json_atomic(run / 'plan.json', plan)
        before = (run / 'run.json').read_bytes()
        for label, operation in (
            ('load', lambda: load_stage_run(run)),
            ('register', lambda: create_job(run, 'outside', task='motion')),
            ('succeed', lambda: update_status(run, 'succeeded'))):
            with self.subTest(operation=label), self.assertRaisesRegex(ValueError, 'fingerprint'):
                operation()
            self.assertEqual((run / 'run.json').read_bytes(), before)


def field_case(field):
    def test(self):
        snap = self.config()
        stage = 'infer' if field == 'tasks' else 'tune'
        plan = foundation.pipeline.build_stage_plan(snap, stage)
        _, run = self.run_record(stage=stage, snapshot=snap, plan=plan)
        index = 1 if field in ('depends_on', 'requires') else 0
        if index:
            create_job(run, **plan['jobs'][0])
        changed = deepcopy(plan['jobs'][index])
        if field == 'task':
            changed['task'] = 'geo'
        elif field == 'tasks':
            changed['tasks'] = changed['tasks'][:-1]
            changed['output']['tasks'] = list(changed['tasks'])
        elif field in ('depends_on', 'requires'):
            changed[field] = []
        elif field == 'input_specs':
            changed[field][-1]['path'] += '.other'
        elif field == 'lineage':
            changed[field]['training_state'] = 'unexpected-state-policy'
        elif field == 'output':
            changed[field]['slot'] = 'wrong-slot'
        self.assert_create_rejected_without_writes(run, changed, 'fields:.*' + field)
    return test


def malformed_case(jobs):
    def test(self):
        _, run, _ = self.small_run(plan_patch={'jobs': deepcopy(jobs)})
        update_status(run, 'running')
        before = (run / 'run.json').read_bytes()
        for label, operation in (
            ('load', lambda: load_stage_run(run)),
            ('register', lambda: create_job(run, 'a', task='motion')),
            ('succeed', lambda: update_status(run, 'succeeded'))):
            with self.subTest(operation=label), self.assertRaises(ValueError):
                operation()
            self.assertEqual((run / 'run.json').read_bytes(), before)
        self.assertEqual(list((run / 'jobs').iterdir()), [])
    return test


def unfinished_case(status):
    def test(self):
        _, run, plan = self.small_run()
        update_status(run, 'running')
        jobs = [create_job(run, **item) for item in plan['jobs']]
        update_status(jobs[0], 'running')
        update_status(jobs[0], 'succeeded')
        if status != 'planned':
            update_status(jobs[1], status)
        self.assert_success_rejected_without_writes(run, 'All child records must succeed')
    return test


for field in ('task', 'tasks', 'depends_on', 'requires', 'input_specs', 'lineage', 'output'):
    setattr(PlanConstraintTests, 'test_frozen_changed_' + field, field_case(field))
for name, jobs in (
    ('jobs_not_list', {}), ('entry_not_object', ['a']), ('missing_id', [{}]),
    ('invalid_id', [{'job_id': '../escape'}]), ('duplicate_ids', [{'job_id': 'a'}, {'job_id': 'a'}])):
    setattr(PlanConstraintTests, 'test_frozen_malformed_' + name, malformed_case(jobs))
for status in ('planned', 'running', 'failed', 'cancelled'):
    setattr(PlanConstraintTests, 'test_frozen_unfinished_' + status, unfinished_case(status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True, type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve(strict=True)
    if not result_dir.is_relative_to((foundation.REPO / 'smoke/results').resolve(strict=True)):
        parser.error('Results must remain below smoke/results')
    if (result_dir / 'tests.json').exists() or (result_dir / 'cases').exists():
        parser.error('Use a fresh result directory; prior evidence is never overwritten')
    foundation.RESULT = result_dir
    names = sorted(name for name in vars(PlanConstraintTests) if name.startswith('test_'))
    suite = unittest.TestSuite(PlanConstraintTests(name) for name in names)
    started = time.time()
    with (result_dir / 'tests.log').open('x', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=foundation.JsonResult).run(suite)
    forbidden = sorted(set(sys.modules).intersection(foundation.BLOCKED_IMPORTS))
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'successful': result.wasSuccessful() and not forbidden, 'elapsed_seconds': time.time()-started,
               'python': sys.version, 'executable': sys.executable, 'yaml': foundation.yaml.__version__,
               'cuda_visible_devices': foundation.os.environ['CUDA_VISIBLE_DEVICES'],
               'forbidden_imports': forbidden, 'events': result.events}
    write_json_atomic(result_dir / 'tests.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'events'}, indent=2))
    if not summary['successful']:
        print((result_dir / 'tests.log').read_text()[-24000:])
    return 0 if summary['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
