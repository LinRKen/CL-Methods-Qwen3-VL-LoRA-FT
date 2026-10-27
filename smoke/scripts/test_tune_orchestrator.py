"""CPU acceptance tests for real Multitask/Singletask tune orchestration."""
import argparse, importlib.abc, json, os, signal, subprocess, sys, time, unittest
from pathlib import Path
from unittest.mock import patch
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LIB = REPO / 'scripts/experiments'
SELF = Path(__file__).resolve()
REAL_POPEN = subprocess.Popen

class Block(importlib.abc.MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'wandb', 'deepspeed', 'vllm'}:
            raise ImportError('CPU test blocked model import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, str(LIB))
import yaml, pipeline, run_tune
from lib import tune_executor as ex, tune_orchestrator as orch
from lib.artifacts import check_manifest
from lib.common import TASKS, read_json, write_json_atomic
from lib.config import load_config, deep_merge
from lib.records import load_experiment, load_job, record_status
RESULT = None

def identity(config, env, group):
    ids = ['GPU-CPU-SIM-' + str(i) for i in group]
    env['CUDA_VISIBLE_DEVICES'] = ','.join(ids)
    for k in ('WANDB_X_STATS_GPU_DEVICE_IDS', 'WANDB__STATS_GPU_DEVICE_IDS'):
        env[k] = json.dumps(group)
    return {'physical_ids': list(group), 'uuids': ids}

def fake_job(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--job-dir', required=True)
    p.add_argument('--gpu-group-index', type=int, default=0)
    p.add_argument('--input', action='append', default=[])
    a = p.parse_args(argv)
    job = Path(a.job_dir)
    mode = json.loads(os.environ.get('SMOKE_JOB_MODES', '{}')).get(job.name, 'success')
    event = {'pid': os.getpid(), 'started': time.monotonic(), 'mode': mode, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json_atomic(job / 'cpu-event.json', event)

    def worker(command, **kw):
        if mode == 'abrupt':
            os._exit(19)
        time.sleep(0.8 if mode == 'slow' else 0.12)
        start = next((i for i, x in enumerate(command) if x.endswith('/train_qwen.py'))) + 1
        env = dict(kw['env'])
        env['SMOKE_FAKE_MODE'] = 'failure' if mode == 'failure' else 'success'
        return REAL_POPEN([sys.executable, '-B', str(REPO / 'smoke/scripts/test_tune_executor.py'), '--fake-worker', *command[start:]], **{**kw, 'env': env})
    try:
        with patch.object(ex, '_gpu_identity', side_effect=identity), patch.object(ex.subprocess, 'Popen', side_effect=worker):
            ex.execute_tune_job(job, gpu_group_index=a.gpu_group_index, input_manifests={k: read_json(v) for k,v in (x.split('=', 1) for x in a.input)})
        if mode == 'corrupt':
            out = Path(load_job(job)[0]['output_dir']) / 'adapter/adapter_model.safetensors'
            out.write_bytes(b'corrupt')
        if mode == 'cleanup':
            r = read_json(job / 'execution.json')
            r['cleanup_error'] = 'simulated'
            write_json_atomic(job / 'execution.json', r)
            return 19
        return 0
    except Exception as e:
        event['error'] = str(e)
        return 17
    finally:
        event.update(finished=time.monotonic(), status=record_status(job)['status'])
        write_json_atomic(job / 'cpu-event.json', event)

def dispatch(command, **kw):
    return REAL_POPEN([command[0], '-B', str(SELF), '--fake-job', *command[2:]], **kw)

def fake_high(argv):
    with patch.object(orch.subprocess, 'Popen', side_effect=dispatch):
        return run_tune.main(argv)

class T(unittest.TestCase):

    def setUp(self):
        self.case = RESULT / 'cases' / self._testMethodName
        self.case.mkdir(parents=True)
        self.n = 0
        base = self.case / 'base'
        base.mkdir()
        (base / 'config.json').write_text('{}')
        (base / 'preprocessor_config.json').write_text('{}')
        data = self.case / 'data'
        for task in (*TASKS, 'whole'):
            (data / task).mkdir(parents=True)
            (data / task / 'train.json').write_text('[]')
        ds = self.case / 'ds.json'
        ds.write_text('{}')
        self.doc = {'experiment': {'name': 'smoke', 'setting': 'singletask', 'base_model': str(base), 'processor_path': str(base), 'dataset_root': str(data), **{k: str(self.case / k) for k in ('records_root', 'checkpoints_root', 'infer_root', 'eval_root')}}, 'tune': {'params': {'deepspeed': str(ds), 'lora_r': 8, 'lora_alpha': 16}, 'run': {'python': sys.executable, 'gpu_groups': [[1]], 'wandb_mode': 'offline', 'master_port': 29901}}}

    def config(self, setting='singletask', extra=None):
        self.n += 1
        doc = deep_merge(self.doc, {'experiment': {'setting': setting}})
        if extra:
            doc = deep_merge(doc, extra)
        p = self.case / ('s' + str(self.n) + '.yaml')
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        return p

    def prepare(self, setting='singletask', extra=None):
        p = self.config(setting, extra)
        snap = load_config(p)
        exp = pipeline.prepare_records(snap, 'exp-' + str(self.n))
        rec, _ = load_experiment(exp)
        run = exp / next((x['path'] for x in rec['stage_runs'] if x['stage'] == 'tune'))
        return (exp, run, p)

    def execute(self, run, modes=None):
        with patch.dict(os.environ, {'SMOKE_JOB_MODES': json.dumps(modes or {})}), patch.object(orch.subprocess, 'Popen', side_effect=dispatch):
            return orch.execute_tune_run(run)

    def events(self, run):
        return {p.parent.name: read_json(p) for p in (run / 'jobs').glob('*/cpu-event.json')}

    def peak(self, run):
        points = []
        for e in self.events(run).values():
            self.assertIn('finished', e)
            points.extend([(e['started'], 1), (e['finished'], -1)])
        n = peak = 0
        for _, d in sorted(points):
            n += d
            peak = max(peak, n)
        return peak

    def reject(self, run):
        with patch.object(orch.subprocess, 'Popen') as spawn, self.assertRaises((ValueError, OSError, KeyError)):
            orch.execute_tune_run(run)
        spawn.assert_not_called()
        self.assertFalse((run / 'orchestrations').exists())

    def test_cpu_import_and_syntax(self):
        self.assertFalse(set(sys.modules) & {'torch', 'transformers', 'peft', 'wandb', 'deepspeed', 'vllm'})
        for x in ('run_tune.py', 'lib/tune_orchestrator.py'):
            compile((LIB / x).read_text(), x, 'exec')

    def test_multitask_runs_one_whole_and_keeps_other_stages_planned(self):
        exp, run, _ = self.prepare('multitask')
        s = self.execute(run)
        self.assertEqual((s['status'], [x['task'] for x in s['jobs']]), ('succeeded', ['whole']))
        self.assertEqual(read_json(run / 'tune-summary.json'), json.loads(json.dumps(s)))
        self.assertEqual(record_status(exp)['status'], 'active')
        self.assertEqual(s['jobs'][0]['adapter']['slot'], 'whole')
        check_manifest(read_json(s['jobs'][0]['adapter']['manifest']), True)
        self.assertEqual([record_status(exp / x['path'])['status'] for x in record_status(exp)['stage_runs'] if x['stage'] != 'tune'], ['planned', 'planned'])

    def test_singletask_runs_six_independent_base_adapters_in_order(self):
        _, run, _ = self.prepare()
        s = self.execute(run)
        self.assertEqual(s['status'], 'succeeded')
        self.assertEqual([x['task'] for x in s['jobs']], list(TASKS))
        self.assertEqual(len({x['adapter']['path'] for x in s['jobs']}), 6)
        self.assertEqual(self.peak(run), 1)
        for row in s['jobs']:
            j, _ = load_job(run / 'jobs' / row['job_id'])
            self.assertEqual((j['training_state_policy'], j['depends_on']), (ex.FRESH, []))
            r = read_json(run / 'jobs' / row['job_id'] / 'execution.json')
            self.assertNotIn('--adapter_name_or_path', r['command'])
            self.assertNotIn('--resume_checkpoint_path', r['command'])
            self.assertEqual(row['adapter']['slot'], row['task'])

    def test_tune_only_marks_experiment_succeeded(self):
        exp, run, _ = self.prepare('multitask', {'experiment': {'enabled_stages': ['tune']}})
        self.execute(run)
        self.assertEqual(record_status(exp)['status'], 'succeeded')

    def test_disjoint_gpu_groups_allow_parallel_jobs(self):
        _, run, _ = self.prepare(extra={'tune': {'run': {'max_parallel_jobs': 2, 'gpu_groups': [[1], [2]]}}})
        self.assertEqual(self.execute(run)['status'], 'succeeded')
        self.assertEqual(self.peak(run), 2)

    def test_same_gpu_group_serializes(self):
        jobs = {f'tune-{t}': {'run': {'gpu_groups': [[1]], 'max_parallel_jobs': 1}} for t in TASKS}
        _, run, _ = self.prepare(extra={'tune': {'run': {'max_parallel_jobs': 2, 'gpu_groups': [[1], [2]]}, 'jobs': jobs}})
        self.execute(run)
        self.assertEqual(self.peak(run), 1)

    def test_port_collision_serializes_disjoint_gpu_groups(self):
        jobs = {f'tune-{t}': {'run': {'gpu_groups': [[1 if i % 2 == 0 else 2]], 'max_parallel_jobs': 1}} for i, t in enumerate(TASKS)}
        _, run, _ = self.prepare(extra={'tune': {'run': {'max_parallel_jobs': 2, 'gpu_groups': [[1], [2]]}, 'jobs': jobs}})
        self.execute(run)
        self.assertEqual(self.peak(run), 1)

    def test_job_override_reaches_low_executor(self):
        x = {'tune': {'run': {'gpu_groups': [[1], [2]], 'max_parallel_jobs': 2}, 'jobs': {'tune-hypo': {'params': {'learning_rate': 0.0007, 'seed': 73}, 'run': {'gpu_groups': [[3]], 'max_parallel_jobs': 1, 'master_port': 29911, 'wandb_project': 'SmokeOverride'}}}}}
        _, run, _ = self.prepare(extra=x)
        s = self.execute(run)
        r = read_json(run / 'jobs/tune-hypo/execution.json')
        cmd = r['command']
        self.assertEqual(cmd[cmd.index('--learning_rate') + 1], '0.0007')
        self.assertEqual(cmd[cmd.index('--seed') + 1], '73')
        self.assertEqual(r['gpu_identity']['physical_ids'], [3])
        self.assertEqual(r['environment']['WANDB_PROJECT'], 'SmokeOverride')
        self.assertEqual(s['jobs'][1]['assignment']['physical_gpu_ids'], [3])

    def test_fail_fast_cancels_pending(self):
        _, run, _ = self.prepare()
        s = self.execute(run, {'tune-motion': 'failure'})
        self.assertEqual(s['status'], 'failed')
        self.assertEqual(len(self.events(run)), 1)
        self.assertEqual([x['status'] for x in s['jobs']], ['failed'] + ['cancelled'] * 5)

    def test_non_fail_fast_continues_independent_jobs(self):
        _, run, _ = self.prepare(extra={'tune': {'run': {'fail_fast': False}}})
        s = self.execute(run, {'tune-motion': 'failure'})
        self.assertEqual(s['status'], 'failed')
        self.assertEqual([x['status'] for x in s['jobs']], ['failed'] + ['succeeded'] * 5)

    def test_fail_fast_stops_and_collects_active_workers(self):
        _, run, _ = self.prepare(extra={'tune': {'run': {'max_parallel_jobs': 2, 'gpu_groups': [[1], [2]]}}})
        s = self.execute(run, {'tune-motion': 'failure', 'tune-hypo': 'slow'})
        self.assertEqual(s['status'], 'failed')
        self.assertTrue(all((x['status'] in ('failed', 'cancelled', 'succeeded') for x in s['jobs'])))

    def test_abrupt_worker_stops_scheduler_even_without_fail_fast(self):
        _, run, _ = self.prepare(extra={'tune': {'run': {'fail_fast': False}}})
        s = self.execute(run, {'tune-motion': 'abrupt'})
        self.assertEqual(s['status'], 'failed')
        self.assertEqual(len(self.events(run)), 1)
        self.assertTrue(all((x['status'] == 'cancelled' for x in s['jobs'][1:])))

    def test_cleanup_error_stops_scheduler(self):
        _, run, _ = self.prepare(extra={'tune': {'run': {'fail_fast': False}}})
        s = self.execute(run, {'tune-motion': 'cleanup'})
        self.assertEqual(s['status'], 'failed')
        self.assertEqual(len(self.events(run)), 1)
        self.assertIn('cleanup', str(s['details']))

    def test_adapter_corruption_rejects_success(self):
        _, run, _ = self.prepare('multitask')
        s = self.execute(run, {'tune-whole': 'corrupt'})
        self.assertEqual(s['status'], 'failed')
        self.assertIn('adapter audit failed', str(s['details']))

    def test_existing_verified_success_is_skipped(self):
        _, run, _ = self.prepare()
        job = run / 'jobs/tune-motion'
        self.assertEqual(fake_job(['--job-dir', str(job)]), 0)
        before = (job / 'execution.json').read_bytes()
        s = self.execute(run)
        self.assertEqual(s['status'], 'succeeded')
        self.assertEqual((job / 'execution.json').read_bytes(), before)
        self.assertIsNone(s['jobs'][0]['log'])
        self.assertEqual(len(read_json(Path(s['invocation_dir']) / 'execution.json')['launches']), 5)

    def test_existing_corrupt_success_is_rejected(self):
        _, run, _ = self.prepare()
        job = run / 'jobs/tune-motion'
        fake_job(['--job-dir', str(job)])
        out = Path(load_job(job)[0]['output_dir']) / 'adapter/adapter_model.safetensors'
        out.write_bytes(b'x')
        self.reject(run)

    def test_terminal_stage_cannot_rerun(self):
        _, run, _ = self.prepare('multitask')
        self.execute(run)
        with self.assertRaisesRegex(ValueError, 'active child'):
            orch.execute_tune_run(run)
        self.assertEqual(len(list((run / 'orchestrations').iterdir())), 1)

    def test_receipt_or_output_conflicts_are_preserved(self):
        _, run, _ = self.prepare('multitask')
        (run / 'jobs/tune-whole/execution.json').write_text('{}')
        self.reject(run)
        _, run, _ = self.prepare('multitask')
        out = Path(load_job(run / 'jobs/tune-whole')[0]['output_dir'])
        out.mkdir(parents=True)
        (out / 'sentinel').write_text('keep')
        self.reject(run)
        self.assertEqual((out / 'sentinel').read_text(), 'keep')

    def test_running_or_failed_jobs_are_not_retried(self):
        for state in ('running', 'failed', 'cancelled'):
            _, run, _ = self.prepare('multitask')
            p = run / 'jobs/tune-whole/job.json'
            r = read_json(p)
            r['status'] = state
            write_json_atomic(p, r)
            self.reject(run)

    def test_frozen_plan_job_and_config_tampering_rejected(self):
        _, run, _ = self.prepare()
        p = run / 'plan.json'
        r = read_json(p)
        r['jobs'][0]['task'] = 'geo'
        write_json_atomic(p, r)
        self.reject(run)
        _, run, _ = self.prepare()
        p = run / 'jobs/tune-motion/job.json'
        r = read_json(p)
        r['task'] = 'geo'
        write_json_atomic(p, r)
        self.reject(run)

    def test_concurrent_high_level_lock_rejects_duplicate(self):
        _, run, _ = self.prepare('multitask')
        with orch._run_lock(run), self.assertRaisesRegex(ValueError, 'Another high-level'):
            orch.execute_tune_run(run)
        self.assertFalse((run / 'orchestrations').exists())

    def test_cli_rejects_unsupported_and_disabled_tune(self):
        for setting in ('transfer_round1', 'zero_shot'):
            p = self.config(setting, {'experiment': {'enabled_stages': ['infer', 'eval']}} if setting == 'zero_shot' else None)
            q = subprocess.run([sys.executable, '-B', str(LIB / 'run_tune.py'), '--config', str(p)], capture_output=True)
            self.assertEqual(q.returncode, 1)
            self.assertIn(b'supports only', q.stderr)
        p = self.config('multitask', {'experiment': {'enabled_stages': ['infer', 'eval']}})
        q = subprocess.run([sys.executable, '-B', str(LIB / 'run_tune.py'), '--config', str(p)], capture_output=True)
        self.assertEqual(q.returncode, 1)
        self.assertIn(b'enable the tune', q.stderr)
        self.assertFalse((self.case / 'records_root').exists())

    def test_cli_frozen_run_rejects_config_overrides_and_relative_path(self):
        _, run, _ = self.prepare('multitask')
        for extra in (['--set', 'tune.params.seed=9'], ['--experiment-id', 'x'], ['--default', str(self.case / 'd.yaml')]):
            q = subprocess.run([sys.executable, '-B', str(LIB / 'run_tune.py'), '--run-dir', str(run), *extra], capture_output=True)
            self.assertEqual(q.returncode, 2)
        q = subprocess.run([sys.executable, '-B', str(LIB / 'run_tune.py'), '--run-dir', 'relative'], capture_output=True)
        self.assertEqual(q.returncode, 1)
        self.assertIn(b'absolute', q.stderr)

    def test_cli_config_freezes_all_stages_executes_only_tune(self):
        p = self.config('multitask')
        snap = load_config(p)
        q = subprocess.run([sys.executable, '-B', str(SELF), '--fake-high', '--config', str(p), '--set', 'tune.params.seed=97', '--experiment-id', 'cli'], capture_output=True, text=True)
        self.assertEqual(q.returncode, 0, q.stderr)
        exp = Path(snap['config']['experiment']['records_root']) / 'cli'
        r, _ = load_experiment(exp)
        runs = {x['stage']: exp / x['path'] for x in r['stage_runs']}
        self.assertEqual(record_status(runs['tune'])['status'], 'succeeded')
        self.assertEqual([record_status(runs[x])['status'] for x in ('infer', 'eval')], ['planned', 'planned'])
        self.assertEqual(load_job(runs['tune'] / 'jobs/tune-whole')[1]['config']['tune']['params']['seed'], 97)

    def test_cli_uses_frozen_snapshot_after_source_yaml_changes(self):
        _, run, p = self.prepare('multitask')
        p.write_text('invalid: [')
        q = subprocess.run([sys.executable, '-B', str(SELF), '--fake-high', '--run-dir', str(run)], capture_output=True, text=True)
        self.assertEqual(q.returncode, 0, q.stderr)

    def test_cli_failure_returns_one(self):
        _, run, _ = self.prepare('multitask')
        q = subprocess.run([sys.executable, '-B', str(SELF), '--fake-high', '--run-dir', str(run)], capture_output=True, env={**os.environ, 'SMOKE_JOB_MODES': json.dumps({'tune-whole': 'failure'})})
        self.assertEqual(q.returncode, 1)
        self.assertEqual(record_status(run)['status'], 'failed')

    def test_sigterm_cancels_and_joins_children(self):
        _, run, _ = self.prepare()
        log = (self.case / 'signal.log').open('xb')
        p = REAL_POPEN([sys.executable, '-B', str(SELF), '--fake-high', '--run-dir', str(run)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'SMOKE_JOB_MODES': json.dumps({'tune-motion': 'slow'})})
        try:
            end = time.monotonic() + 15
            while not (run / 'jobs/tune-motion/cpu-event.json').exists():
                if time.monotonic() > end or p.poll() is not None:
                    self.fail('job did not start')
                time.sleep(0.02)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=20), 130)
        finally:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=20)
            log.close()
        self.assertEqual(record_status(run)['status'], 'cancelled')
        for e in self.events(run).values():
            with self.assertRaises(ProcessLookupError):
                os.kill(e['pid'], 0)

def main():
    global RESULT
    p = argparse.ArgumentParser()
    p.add_argument('--result-dir', required=True)
    a = p.parse_args()
    RESULT = Path(a.result_dir).resolve(strict=True)
    if not RESULT.is_relative_to((REPO / 'smoke/results').resolve(strict=True)):
        p.error('result directory must be under smoke/results')
    if (RESULT / 'cases').exists() or (RESULT / 'tests.json').exists():
        p.error('refusing existing evidence')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    write_json_atomic(RESULT / 'tests.json', {'tests': r.testsRun, 'failures': len(r.failures), 'errors': len(r.errors), 'successful': r.wasSuccessful()})
    return 0 if r.wasSuccessful() else 1
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fake-job':
        raise SystemExit(fake_job(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == '--fake-high':
        raise SystemExit(fake_high(sys.argv[2:]))
    raise SystemExit(main())
