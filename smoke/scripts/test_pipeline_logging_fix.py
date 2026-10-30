"""CPU regression for logging before frozen-plan validation.

Only synthetic records below smoke/results are used. No workers or GPUs run.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.abc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['GIT_OPTIONAL_LOCKS'] = '0'
sys.dont_write_bytecode = True
BLOCKED = {'torch', 'vllm', 'transformers', 'deepspeed', 'wandb', 'peft', 'qwen_vl_utils'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ImportError('CPU logging regression forbids model imports: ' + fullname)


sys.meta_path.insert(0, NoModels())
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experiments'))
import yaml
from lib import orchestration_logging as logging
from lib.common import fingerprint, read_json, write_json_atomic
from lib.config import load_config
from lib.records import create_experiment, create_stage_run, load_stage_run, load_stage_run_metadata
from lib.tune_orchestrator import execute_tune_run
from lib.infer_orchestrator import execute_infer_run
from lib.eval_orchestrator import execute_eval_run

EXECUTE = {'tune': execute_tune_run, 'infer': execute_infer_run, 'eval': execute_eval_run}
RESULT = None
AUDITS = []


def tree(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in directory.rglob('*') if p.is_file()}


class LoggingChecks:
    def setUp(self):
        self.case = RESULT / 'cases' / self.stage / self._testMethodName
        self.case.mkdir(parents=True)
        experiment = {key: str(self.case / key) for key in (
            'base_model', 'processor_path', 'judge_model', 'dataset_root',
            'records_root', 'checkpoints_root', 'infer_root', 'eval_root', 'logs_root')}
        experiment.update(name='logging-fix-cpu', setting='multitask')
        path = self.case / 'config.yaml'
        path.write_text(yaml.safe_dump({'experiment': experiment}))
        self.snapshot = load_config(path)
        exp = create_experiment(self.snapshot, 'cpu-case')
        plan = {'stage': self.stage, 'config_fingerprint': self.snapshot['config_fingerprint'],
                'jobs': [{'job_id': 'first'}]}
        self.run = create_stage_run(exp, self.stage, run_id='cpu-' + self.stage, stage_plan=plan)
        record, self.saved_snapshot = load_stage_run_metadata(self.run)
        self.logs = (Path(experiment['logs_root']) / record['experiment_id'] / 'stage-runs' /
                     self.stage / record['run_id'] / 'orchestrations')

    def invocation(self, status, error_type=None):
        folders = sorted(self.logs.iterdir())
        self.assertEqual(len(folders), 1)
        folder = folders[0]
        receipt = read_json(folder / 'invocation.json')
        self.assertEqual(receipt['stage'], self.stage)
        self.assertEqual(receipt['status'], status)
        self.assertEqual(receipt['run_dir'], str(self.run.resolve()))
        self.assertEqual(receipt['config_fingerprint'], self.saved_snapshot['config_fingerprint'])
        self.assertTrue(receipt['finished_at'])
        log = (folder / 'orchestrator.log').read_text()
        self.assertIn('Orchestration logs:', log)
        if status != 'succeeded':
            failure = read_json(folder / 'failure.json')
            self.assertEqual(failure['status'], status)
            self.assertEqual(failure['error_type'], error_type)
            self.assertTrue(failure['traceback'])
            self.assertIn(failure['message'], log)
        else:
            self.assertFalse((folder / 'failure.json').exists())
        AUDITS.append({'test': self.id(), 'invocation_dir': str(folder), 'status': status})
        return folder, log

    def reject_public(self, error):
        before = tree(self.run)
        with patch('subprocess.Popen') as spawn, patch('subprocess.run') as command:
            with self.assertRaises(error):
                EXECUTE[self.stage](self.run)
            spawn.assert_not_called()
            command.assert_not_called()
        self.assertEqual(before, tree(self.run))
        self.invocation('failed', error.__name__)

    def corrupt_plan(self):
        plan = read_json(self.run / 'plan.json')
        plan['jobs'].append({'job_id': 'extra'})
        write_json_atomic(self.run / 'plan.json', plan)

    def test_frozen_plan_tampering_logs_failure_without_dispatch(self):
        self.corrupt_plan()
        self.reject_public(ValueError)

    def test_missing_plan_logs_failure_without_dispatch(self):
        (self.run / 'plan.json').unlink()
        self.reject_public(FileNotFoundError)

    def test_malformed_plan_json_logs_failure_without_dispatch(self):
        (self.run / 'plan.json').write_text('{broken')
        self.reject_public(json.JSONDecodeError)

    def test_duplicate_plan_ids_still_rejected_and_logged(self):
        plan = read_json(self.run / 'plan.json')
        plan['jobs'].append(deepcopy(plan['jobs'][0]))
        write_json_atomic(self.run / 'plan.json', plan)
        record = read_json(self.run / 'run.json')
        record['plan_fingerprint'] = fingerprint(plan)
        write_json_atomic(self.run / 'run.json', record)
        self.reject_public(ValueError)

    def test_stage_mismatch_logged_before_callback(self):
        callback = Mock()
        wrong = {'tune': 'infer', 'infer': 'eval', 'eval': 'tune'}[self.stage]
        before = tree(self.run)
        with self.assertRaisesRegex(ValueError, 'registered .* stage run'):
            logging.run_logged(wrong, self.run, callback)
        callback.assert_not_called()
        self.assertEqual(before, tree(self.run))
        own = self.stage
        self.stage = wrong
        self.logs = self.logs.parents[2] / wrong / ('cpu-' + own) / 'orchestrations'
        try:
            self.invocation('failed', 'ValueError')
        finally:
            self.stage = own

    def test_run_metadata_changed_during_log_creation_rejected(self):
        original = logging._new_invocation
        callback = Mock()
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            record = read_json(self.run / 'run.json')
            record['updated_at'] = 'SMOKE-CHANGED-DURING-INITIALIZATION'
            write_json_atomic(self.run / 'run.json', record)
            return value
        with patch.object(logging, '_new_invocation', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'metadata changed'):
                logging.run_logged(self.stage, self.run, callback)
        callback.assert_not_called()
        self.invocation('failed', 'RuntimeError')

    def test_success_logs_both_streams_and_returns_result(self):
        before = tree(self.run)
        def callback(run, invocation):
            self.assertEqual(run, self.run.resolve())
            self.assertEqual(read_json(invocation / 'invocation.json')['status'], 'running')
            print('SMOKE-STDOUT')
            print('SMOKE-STDERR', file=sys.stderr)
            return {'status': 'succeeded', 'invocation_dir': str(invocation)}
        result = logging.run_logged(self.stage, self.run, callback)
        folder, log = self.invocation('succeeded')
        self.assertEqual(result['invocation_dir'], str(folder))
        self.assertIn('SMOKE-STDOUT', log)
        self.assertIn('SMOKE-STDERR', log)
        self.assertEqual(before, tree(self.run))

    def test_valid_snapshot_replaced_during_log_creation_rejected(self):
        original = logging._new_invocation
        callback = Mock()
        before_record = read_json(self.run / 'run.json')
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            path = self.run / 'config.snapshot.json'
            snapshot = read_json(path)
            snapshot['sources'][0]['sha256'] = 'f' * 64
            snapshot['snapshot_fingerprint'] = fingerprint({
                key: item for key, item in snapshot.items() if key != 'snapshot_fingerprint'})
            write_json_atomic(path, snapshot)
            record, validated = load_stage_run(self.run)
            self.assertEqual(record, before_record)
            self.assertEqual(validated['config_fingerprint'], self.saved_snapshot['config_fingerprint'])
            self.assertNotEqual(validated, self.saved_snapshot)
            return value
        with patch.object(logging, '_new_invocation', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'metadata changed'):
                logging.run_logged(self.stage, self.run, callback)
        callback.assert_not_called()
        self.invocation('failed', 'RuntimeError')

    def test_execution_exception_retains_error_and_reraises(self):
        before = tree(self.run)
        with self.assertRaisesRegex(RuntimeError, 'SMOKE-EXECUTION-ERROR'):
            logging.run_logged(self.stage, self.run, Mock(side_effect=RuntimeError('SMOKE-EXECUTION-ERROR')))
        self.invocation('failed', 'RuntimeError')
        self.assertEqual(before, tree(self.run))

    def test_keyboard_interrupt_records_cancelled(self):
        before = tree(self.run)
        with self.assertRaises(KeyboardInterrupt):
            logging.run_logged(self.stage, self.run, Mock(side_effect=KeyboardInterrupt('SMOKE-INTERRUPT')))
        self.invocation('cancelled', 'KeyboardInterrupt')
        self.assertEqual(before, tree(self.run))

    def test_metadata_loader_does_not_weaken_strict_loader(self):
        self.corrupt_plan()
        before = tree(self.run)
        record, snapshot = load_stage_run_metadata(self.run)
        self.assertEqual(record['stage'], self.stage)
        self.assertEqual(snapshot, self.saved_snapshot)
        with self.assertRaises(ValueError):
            load_stage_run(self.run)
        self.assertFalse(self.logs.exists())
        self.assertEqual(before, tree(self.run))

    def test_real_cli_preflight_returns_one_and_keeps_failure_files(self):
        self.corrupt_plan()
        before = tree(self.run)
        command = [sys.executable, '-B', str(REPO / 'scripts/experiments' / ('run_' + self.stage + '.py')),
                   '--run-dir', str(self.run)]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=35, env=dict(os.environ))
        (self.case / 'cli.stdout').write_text(completed.stdout)
        (self.case / 'cli.stderr').write_text(completed.stderr)
        write_json_atomic(self.case / 'cli.json', {'command': command, 'exit_code': completed.returncode,
                                                  'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
        self.assertEqual(completed.returncode, 1)
        self.assertIn('fingerprint mismatch', completed.stderr)
        self.invocation('failed', 'ValueError')
        self.assertEqual(before, tree(self.run))
        self.assertFalse(list(self.case.rglob('execution.json')))

    def test_corrupt_snapshot_rejected_before_trusting_logs_root(self):
        path = self.run / 'config.snapshot.json'
        snapshot = read_json(path)
        snapshot['config']['infer']['params']['seed'] += 1
        write_json_atomic(path, snapshot)
        before = tree(self.run)
        callback = Mock()
        with self.assertRaises(ValueError):
            logging.run_logged(self.stage, self.run, callback)
        callback.assert_not_called()
        self.assertFalse(self.logs.exists())
        self.assertEqual(before, tree(self.run))


def main():
    global RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', required=True, type=Path)
    parser.add_argument('--test', action='append', help='Run only the named method for all three stages')
    args = parser.parse_args()
    RESULT = args.result_root.resolve()
    if not RESULT.is_relative_to((REPO / 'smoke/results').resolve(strict=True)):
        parser.error('results must remain below smoke/results')
    RESULT.mkdir(parents=True, exist_ok=False)
    suite = unittest.TestSuite()
    for stage in EXECUTE:
        case_type = type(stage.title() + 'LoggingTests', (LoggingChecks, unittest.TestCase), {'stage': stage})
        if args.test:
            suite.addTests(case_type(name) for name in args.test)
        else:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case_type))
    started = time.monotonic()
    with (RESULT / 'tests.log').open('x') as stream, (RESULT / 'production.log').open('x') as production:
        with redirect_stdout(production), redirect_stderr(production):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    imports = sorted(BLOCKED.intersection(sys.modules))
    summary = {'tests': result.testsRun, 'selected_tests': args.test,
               'failures': len(result.failures), 'errors': len(result.errors),
               'successful': result.wasSuccessful() and not imports,
               'elapsed_seconds': time.monotonic() - started, 'python': sys.executable,
               'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
               'forbidden_imports': imports, 'audited_invocations': AUDITS,
               'details': [{'test': str(test), 'traceback': value} for test, value in result.failures + result.errors]}
    write_json_atomic(RESULT / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('details', 'audited_invocations')}, indent=2))
    if not summary['successful']:
        print((RESULT / 'tests.log').read_text())
    return 0 if summary['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
