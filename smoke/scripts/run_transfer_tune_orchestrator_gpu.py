"""Exercise all 36 transfer tune jobs on 4B, physical GPU 1 only."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LIB = REPO / 'scripts/experiments'
sys.path.insert(0, str(LIB))
import yaml
from lib.artifacts import check_manifest
from lib.common import TASKS, read_json, write_json_atomic
from lib.config import load_config
from lib.records import load_experiment, load_job, record_status
from lib.tune_executor import FRESH, CONTINUE

BASE = Path('/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct')
DATA = Path('/home/linruikang/data/datasets/COSINE')
PYTHON = '/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python'
UUID = 'GPU-63a25b73-5c01-919f-188a-29c116904574'


def gpu_query():
    def query(kind, columns):
        command = ['nvidia-smi', '-i', '1', '--query-' + kind + '=' + columns,
                   '--format=csv,noheader,nounits']
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
        return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    cards = query('gpu', 'index,uuid,memory.used,memory.total')
    if len(cards) != 1 or cards[0][:2] != ['1', UUID]:
        raise ValueError('GPU 1 identity mismatch: ' + repr(cards))
    processes = query('compute-apps', 'pid,gpu_uuid,used_memory')
    if any(row[1] != UUID for row in processes):
        raise ValueError('Unexpected GPU identity in scoped process query')
    return {'timestamp': time.time(), 'gpu': cards[0], 'processes': processes}


def setup(root):
    processor = root / 'processor'
    processor.mkdir()
    for item in BASE.iterdir():
        if item.is_file() and not item.name.startswith('.') and not any(
                word in item.name for word in ('safetensors', 'pytorch_model', 'optimizer', 'training_args')):
            (processor / item.name).symlink_to(item)
    samples = []
    for task in TASKS:
        source = DATA / task / 'train.json'
        records = json.loads(source.read_text())
        indices = []
        predicates = [lambda x: len(x.get('image', [])) == 1 and not x.get('video'),
                      lambda x: len(x.get('image', [])) == 1 and not x.get('video'),
                      lambda x: len(x.get('image', [])) > 1, lambda x: bool(x.get('video'))]
        for predicate in predicates:
            index = next((i for i, x in enumerate(records) if i not in indices and predicate(x)), None)
            if index is None:
                index = next(i for i in range(len(records)) if i not in indices)
            indices.append(index)
        target = root / 'dataset' / task / 'train.json'
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps([records[i] for i in indices], ensure_ascii=False, indent=2))
        samples.append({'task': task, 'source': str(source), 'source_count': len(records),
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'target': str(target), 'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'indices': indices, 'ids': [records[i].get('_meta', {}).get('id') for i in indices],
                        'image_counts': [len(records[i].get('image', [])) for i in indices],
                        'video_counts': [len(records[i].get('video', [])) for i in indices]})
    write_json_atomic(root / 'samples.json', samples)
    configs = {}
    for setting, port in (('transfer_round1', 29887),):
        # Check only the ports this smoke run will use.
        with socket.socket() as handle:
            handle.bind(('127.0.0.1', port))
        cfg = {'experiment': {'name': 'transfer-tune-orchestrator-smoke', 'setting': setting,
                             'dataset_root': str(root / 'dataset'), 'base_model': str(BASE),
                             'processor_path': str(processor),
                             **{k: str(root / k) for k in ('records_root', 'checkpoints_root', 'infer_root', 'eval_root')}},
               'tune': {'params': {'num_train_epochs': 1, 'per_device_train_batch_size': 1,
                                   'gradient_accumulation_steps': 2, 'lora_r': 8, 'lora_alpha': 16,
                                   'dataloader_num_workers': 0, 'save_strategy': 'steps', 'save_steps': 1,
                                   'save_total_limit': 3, 'report_to': 'wandb'},
                        'run': {'gpu_groups': [[1]], 'max_parallel_jobs': 1, 'master_port': port,
                                'wandb_mode': 'offline', 'wandb_project': 'Qwen3VL-LoRA-FT-Smoke'}}}
        path = root / (setting + '.yaml')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json_atomic(root / (setting + '-config.snapshot.json'), load_config(path))
        configs[setting] = path
    return configs


def train(root, setting, config):
    command = [PYTHON, '-B', str(LIB / 'run_tune.py'), '--config', str(config),
               '--experiment-id', 'smoke-' + setting]
    write_json_atomic(root / (setting + '-command.json'), {'argv': command})
    observations = []
    stop = threading.Event()
    def monitor():
        with (root / (setting + '-gpu.jsonl')).open('x') as output:
            while not stop.is_set():
                try:
                    row = gpu_query()
                except Exception as error:
                    row = {'timestamp': time.time(), 'error': repr(error)}
                observations.append(row)
                output.write(json.dumps(row) + '\n')
                output.flush()
                stop.wait(5)
    thread = threading.Thread(target=monitor, daemon=True)
    started = time.monotonic()
    with (root / (setting + '.log')).open('xb') as output:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1',
                                                                 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
        write_json_atomic(root / (setting + '-pid.json'), {'pid': process.pid})
        thread.start()
        try:
            code = process.wait()
        except BaseException:
            # Only terminate the exact smoke orchestrator; it owns worker cleanup.
            process.terminate()
            process.wait()
            raise
        finally:
            stop.set()
            thread.join(timeout=15)
    evidence = {'returncode': code, 'elapsed_seconds': time.monotonic() - started,
                'observed_max_whole_gpu_mib': max((float(x['gpu'][2]) for x in observations if 'gpu' in x), default=None)}
    write_json_atomic(root / (setting + '-exit.json'), evidence)
    print(setting + ': ' + json.dumps(evidence), flush=True)
    if code:
        raise RuntimeError(setting + ' CLI failed; do not automatically rerun')


def verify(root):
    # Audit existing artifacts without starting training or initializing CUDA.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from safetensors.torch import load_file
    checks = {}
    cases = {}
    all_pids, wandb_ids, outputs = set(), set(), set()
    previous_tensors = None
    previous_row = None
    for setting in ('transfer_round1',):
        exp = root / 'records_root' / ('smoke-' + setting)
        experiment, _ = load_experiment(exp)
        runs = {x['stage']: exp / x['path'] for x in experiment['stage_runs']}
        summary = read_json(runs['tune'] / 'tune-summary.json')
        expected_tasks = list(TASKS) + [target for source in TASKS for target in TASKS if source != target]
        checks[setting + '_expected_tasks'] = [x['task'] for x in summary['jobs']] == expected_tasks
        checks[setting + '_stage_succeeded'] = summary['status'] == record_status(runs['tune'])['status'] == 'succeeded'
        checks[setting + '_infer_eval_not_executed'] = all(record_status(runs[s])['status'] == 'planned' for s in ('infer', 'eval'))
        checks[setting + '_experiment_active'] = experiment['status'] == 'active'
        high_receipt = read_json(Path(summary['invocation_dir']) / 'execution.json')
        all_pids.add(high_receipt['pid'])
        checks[setting + '_launch_count'] = len(high_receipt['launches']) == len(expected_tasks)
        checks[setting + '_sequential_on_gpu1'] = all(x['assignment']['physical_gpu_ids'] == [1] for x in high_receipt['launches'])
        previous_end = None
        rows_by_id = {r['job_id']:r for r in summary['jobs']}
        for row in summary['jobs']:
            parent_id = row['lineage'].get('parent_job')
            previous_row = rows_by_id[parent_id] if parent_id else None
            previous_tensors = (load_file(str(Path(previous_row['adapter']['path'])/'adapter_model.safetensors'), device='cpu') if previous_row else None)
            name = setting + '/' + row['job_id']
            directory = runs['tune'] / 'jobs' / row['job_id']
            job, snapshot = load_job(directory)
            artifacts = {x['role']: check_manifest(read_json(directory / x['path']), verify_contents=True) for x in job['artifacts']}
            receipt = read_json(directory / 'execution.json')
            low_summary = read_json(artifacts['training-summary']['path'])
            state = read_json(Path(artifacts['checkpoint-2']['path']) / 'trainer_state.json')
            first = read_json(Path(artifacts['checkpoint-1']['path']) / 'trainer_state.json')
            checks[name + '_all_artifacts'] = set(artifacts) == {'adapter', 'checkpoint-1', 'checkpoint-2', 'training-summary'}
            checks[name + '_correct_inheritance_policy'] = (
                job['training_state_policy'] == (FRESH if previous_row is None else CONTINUE)
                and not job['resume_training_state']
                and '--resume_checkpoint_path' not in receipt['command']
                and snapshot['config']['experiment']['base_model'] == str(BASE))
            manifest = artifacts['adapter']
            checks[name + '_output_lineage'] = (
                manifest['metadata']['lineage'] == job['lineage']
                and manifest['metadata']['training_state_policy'] == job['training_state_policy'])
            if previous_row is None:
                checks[name + '_base_initialization'] = (
                    not job['depends_on'] and not job['requires']
                    and not any(x['kind'] == 'adapter' for x in job['inputs'].values())
                    and '--adapter_name_or_path' not in receipt['command']
                    and row['parent_adapter'] is None)
            else:
                parent = read_json(previous_row['adapter']['manifest'])
                checks[name + '_exact_parent_binding'] = (
                    [v for v in job['inputs'].values() if v['kind'] == 'adapter'] == [parent]
                    and job['depends_on'] == [previous_row['job_id']]
                    and row['parent_adapter']['producer'] == parent['producer']
                    and row['parent_adapter']['fingerprint'] == parent['fingerprint']
                    and row['parent_adapter']['slot'] == 'round1-'+previous_row['task']
                    and receipt['command'][receipt['command'].index('--adapter_name_or_path')+1] == parent['path']
                    and manifest['metadata']['inherited_inputs'] == {
                        'parent-'+previous_row['task']: parent['fingerprint']})
            checks[name + '_success'] = job['status'] == row['status'] == receipt['status'] == 'succeeded' and receipt['returncode'] == row['returncode'] == 0
            checks[name + '_training_steps'] = (low_summary['global_step'] == state['global_step'] == 2
                and low_summary['epoch'] == state['epoch'] == 1.0 and low_summary['train_samples'] == 4
                and low_summary['world_size'] == 1 and low_summary['trainable_parameters'] == 5898240
                and first['global_step'] == 1 and first['epoch'] == 0.5)
            checkpoint_states = {}
            for step in (1, 2):
                checkpoint = Path(artifacts['checkpoint-' + str(step)]['path'])
                shard = checkpoint / ('global_step' + str(step))
                optimizer = torch.load(next(shard.glob('*optim_states.pt')), map_location='cpu',
                                       weights_only=False)['optimizer_state_dict']['optimizer_state_dict']
                engine = torch.load(next(shard.glob('*model_states.pt')), map_location='cpu',
                                    weights_only=False)
                scheduler = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=False)
                optimizer_steps = [value['step'].item() if isinstance(value['step'], torch.Tensor)
                                   else value['step'] for value in optimizer['state'].values()]
                checkpoint_states[str(step)] = {'optimizer_steps': optimizer_steps,
                    'global_steps': engine['global_steps'], 'global_samples': engine['global_samples'],
                    'scheduler_last_epoch': scheduler['last_epoch'],
                    'scheduler_step_count': scheduler['_step_count'], 'scheduler_last_lr': scheduler['_last_lr']}
                checks[name + '_fresh_optimizer_engine_scheduler_step' + str(step)] = (
                    bool(optimizer_steps) and all(value == step for value in optimizer_steps)
                    and engine['global_steps'] == step and engine['global_samples'] == 2 * step
                    and scheduler['last_epoch'] == step and scheduler['_step_count'] == step + 1
                    and scheduler['_last_lr'] == ([0.0005] if step == 1 else [0.0]))
                del optimizer, engine, scheduler
            logs = [x for x in state['log_history'] if 'loss' in x]
            checks[name + '_finite_training'] = ([x['step'] for x in logs] == [1, 2] and all(
                math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) and x['grad_norm'] > 0 for x in logs))
            checks[name + '_gpu1_and_wandb_selection'] = (receipt['gpu_identity'] == {'physical_ids': [1], 'uuids': [UUID]}
                and receipt['environment']['CUDA_VISIBLE_DEVICES'] == UUID
                and all(receipt['environment'][key] == '[1]' for key in ('WANDB_X_STATS_GPU_DEVICE_IDS', 'WANDB__STATS_GPU_DEVICE_IDS')))
            checks[name + '_summary_adapter_matches'] = row['adapter']['path'] == artifacts['adapter']['path'] and row['adapter']['fingerprint'] == artifacts['adapter']['fingerprint']
            media_path = Path(snapshot['config']['experiment']['dataset_root']) / row['task'] / 'train.json'
            checks[name + '_exact_dataset_binding'] = [x['path'] for x in job['input_specs'] if x['kind'] == 'annotation'] == [str(media_path)]
            tensors = load_file(str(Path(artifacts['adapter']['path']) / 'adapter_model.safetensors'), device='cpu')
            counts = {'all': len(tensors), 'lora_B': sum('lora_B' in k for k in tensors)}
            checks[name + '_valid_adapter'] = counts == {'all': 288, 'lora_B': 144} and all(
                'lora_' in k and 'visual' not in k.split('.') and v.device.type == 'cpu' and bool(torch.isfinite(v).all())
                and ('lora_B' not in k or bool(torch.count_nonzero(v))) for k, v in tensors.items())
            first_tensors = load_file(str(Path(artifacts['checkpoint-1']['path']) / 'adapter_model.safetensors'), device='cpu')
            checks[name + '_scheduler_restarts_from_zero_lr'] = first['log_history'][0]['learning_rate'] == 0.0
            if previous_tensors is None:
                checks[name + '_fresh_lora_b_before_nonzero_lr'] = all(
                    not bool(torch.count_nonzero(v)) for k,v in first_tensors.items() if 'lora_B' in k)
            else:
                checks[name + '_all_288_initial_tensors_equal_parent_final'] = (
                    set(first_tensors) == set(previous_tensors) and all(
                        torch.equal(first_tensors[k],previous_tensors[k]) for k in first_tensors))
            checks[name + '_adapter_updated_by_step2'] = set(tensors) == set(first_tensors) and any(
                not torch.equal(tensors[k], first_tensors[k]) for k in tensors)
            if previous_row:
                checks[name + '_source_not_sibling_parent'] = (job['lineage']['round'] == 2 and job['lineage']['source_task'] == previous_row['task'] and previous_row['lineage']['round'] == 1 and previous_row['task'] != row['task'])
            else:
                checks[name + '_independent_round1'] = job['lineage']['round'] == 1
            del first_tensors
            log = Path(receipt['log_path']).read_text()
            checks[name + '_no_traceback_or_oom'] = 'Traceback (most recent call last)' not in log and 'CUDA out of memory' not in log
            if previous_end is not None:
                checks[name + '_after_previous_job'] = receipt['started_at'] >= previous_end
            previous_end = receipt['finished_at']
            wandb_ids.add(receipt['environment']['WANDB_RUN_ID'])
            wandb_root = Path(receipt['environment']['WANDB_DIR'])
            checks[name + '_one_offline_wandb_run'] = len(list(wandb_root.glob('wandb/offline-run-*'))) == 1
            outputs.add(job['output_dir'])
            all_pids.update((receipt['pid'], receipt['launcher_pid']))
            cases[name] = {'successful': all(v for k, v in checks.items() if k.startswith(name + '_')),
                           'job_dir': str(directory), 'output_dir': job['output_dir'], 'summary': low_summary, 'checkpoint_training_state': checkpoint_states,
                           'losses': logs, 'gpu_identity': receipt['gpu_identity'], 'adapter_tensors': counts,
                           'wandb_run_id': receipt['environment']['WANDB_RUN_ID'], 'elapsed_seconds': receipt['elapsed_seconds']}
        for launch in high_receipt['launches']:
            all_pids.add(launch['pid'])
        monitor_rows = [json.loads(line) for line in (root / (setting + '-gpu.jsonl')).read_text().splitlines()]
        checks[setting + '_gpu_monitor_succeeded'] = bool(monitor_rows) and all('error' not in x for x in monitor_rows)
        for observation in monitor_rows:
            all_pids.update(int(row[0]) for row in observation.get('processes', []))
    for row in read_json(root / 'samples.json'):
        checks['source_unchanged_' + row['task']] = hashlib.sha256(Path(row['source']).read_bytes()).hexdigest() == row['source_sha256']
        checks['sample_unchanged_' + row['task']] = hashlib.sha256(Path(row['target']).read_bytes()).hexdigest() == row['target_sha256']
    checks['36_distinct_outputs_and_wandb_ids'] = len(outputs) == len(wandb_ids) == 36
    checks['cuda_not_initialized_in_audit'] = not torch.cuda.is_initialized()
    live = []
    for pid in sorted(all_pids):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            live.append(pid)
    checks['all_recorded_processes_exited'] = not live
    result = {'successful': all(checks.values()), 'checks': checks, 'cases': cases,
              'all_recorded_pids': sorted(all_pids), 'live_pids': live, 'audit_only': True}
    target = root / 'outcomes.verified.json'
    if target.exists():
        raise ValueError('Refusing to overwrite verification evidence')
    write_json_atomic(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result['successful'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.result_dir.resolve(strict=True)
    if not root.is_relative_to((REPO / 'smoke/results').resolve(strict=True)):
        parser.error('Results must be under smoke/results')
    if args.verify_only:
        return verify(root)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if (root / 'samples.json').exists():
        parser.error('Refusing to reuse evidence or retrain completed jobs')
    before = gpu_query()
    write_json_atomic(root / 'gpu-before.json', before)
    if before['processes']:
        parser.error('GPU 1 is busy; stop without touching existing processes')
    configs = setup(root)
    for setting, config in configs.items():
        train(root, setting, config)
    write_json_atomic(root / 'gpu-after-training.json', gpu_query())
    print('Training completed. Use --verify-only with CUDA_VISIBLE_DEVICES empty.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
