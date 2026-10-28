'''Read-only additional evidence audit; never invokes inference or training.'''
import argparse
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experiments'))
from lib.common import read_json, write_json_atomic, sha256_file
from lib.records import load_job

def audit(root):
    checks = {}; details = {}; pids = set()
    live = [json.loads(line) for line in (root / 'live-process-identity.jsonl').read_text().splitlines() if line.strip()]
    for item in live:
        pids.add(item['pid'])
        cmd = item['cmdline']
        if any(token in cmd for token in ('run_infer_job.py', 'infer_worker.py', 'resource_tracker')):
            checks['live_gpu_1_only:' + str(item['pid'])] = item['environment'].get('CUDA_VISIBLE_DEVICES') == '1' and item['environment'].get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    for case in read_json(root / 'cases.json'):
        name = case['setting']; directory = Path(case['job_dir'])
        job, snapshot = load_job(directory); out = Path(job['output_dir'])
        summary = read_json(out / 'inference_result.json')
        receipt = read_json(directory / 'execution.json')
        log = (out / 'logs/infer.log').read_text(errors='replace')
        counts = {'engine_initializations': log.count('Initializing a V1 LLM engine'), 'model_load_starts': log.count('Starting to load model'), 'task_starts': log.count(' with the shared engine'), 'task_completions': log.count('[4/4] inference complete')}
        for key, expected in [('engine_initializations', 1), ('model_load_starts', 1), ('task_starts', 6), ('task_completions', 6)]:
            checks[name + ':' + key] = counts[key] == expected
        paths = sorted(set(re.findall(r'/tmp/pipeline-vllm-model-[A-Za-z0-9_-]+', log)))
        checks[name + ':single_temporary_model_view'] = len(paths) == 1
        checks[name + ':temporary_model_view_removed'] = bool(paths) and all(not Path(p).exists() for p in paths)
        rows = [json.loads(line) for line in (root / (name + '-gpu.jsonl')).read_text().splitlines() if line.strip()]
        valid = [row for row in rows if 'gpu' in row]
        errors = [row for row in rows if 'error' in row]
        gpu_pids = {int(p[0]) for row in valid for p in row['processes']}
        checks[name + ':monitor_no_errors'] = not errors
        checks[name + ':monitor_all_gpu_1_uuid'] = bool(valid) and all(row['gpu'][:2] == ['1', 'GPU-63a25b73-5c01-919f-188a-29c116904574'] and all(p[1] == row['gpu'][1] for p in row['processes']) for row in valid)
        checks[name + ':monitor_observed_compute'] = bool(gpu_pids)
        checks[name + ':process_uuid_matches_live_tree'] = bool(gpu_pids) and gpu_pids <= {item['pid'] for item in live}
        pids.update(gpu_pids)
        pids.update(int(p) for p in re.findall(r'pid=([0-9]+)', log))
        pids.update(receipt[k] for k in ('pid', 'launcher_pid') if k in receipt)
        pids.add(read_json(root / (name + '-launcher.json'))['pid'])
        checks[name + ':adapter_enabled_in_engine'] = ("'enable_lora': True" in log) == (name == 'multitask')
        checks[name + ':no_runtime_failure'] = not any(marker in log for marker in ('Traceback (most recent call last)', 'CUDA out of memory', 'EngineCore failed'))
        total = sum(t['sample_count'] for t in summary['tasks'])
        checks[name + ':24_predictions'] = total == 24
        details[name] = {**counts, 'temporary_model_views': paths, 'observations': len(valid), 'monitor_errors': errors, 'gpu_pids': sorted(gpu_pids), 'observed_max_whole_gpu_mib': max(float(row['gpu'][2]) for row in valid), 'predictions': total, 'nccl_exit_warning': 'destroy_process_group() was not called' in log, 'worker_elapsed_seconds': summary['elapsed_seconds'], 'cli_elapsed_seconds': read_json(root / (name + '-exit.json'))['elapsed_seconds']}
    for evidence in root.rglob('adoption-evidence.json'):
        doc = read_json(evidence); source = read_json(doc['source_manifest']); source_dir = Path(source['path'])
        checks['adopted_real_weight_unchanged'] = sha256_file(source_dir / 'adapter_model.safetensors') == doc['source_weight_sha256']
        checks['adopted_source_fingerprint_unchanged'] = source['fingerprint'] == doc['source_fingerprint']
        checks['adoption_discloses_no_training'] = doc['training_executed'] is False
        details['adoption'] = doc
    for pid in sorted(pids):
        checks['all_recorded_processes_exited:' + str(pid)] = not Path('/proc/' + str(pid)).exists()
    checks['eight_worker_tree_processes_captured'] = len({item['pid'] for item in live}) >= 8
    report = {'successful': all(checks.values()), 'checks_count': len(checks), 'checks': checks, 'pids': sorted(pids), 'details': details}
    write_json_atomic(root / 'additional-verification.json', report)
    print(json.dumps({'successful': report['successful'], 'checks_count': len(checks), 'pids': report['pids'], 'failed': [k for k, passed in checks.items() if not passed], 'details': details}, ensure_ascii=False, indent=2))
    return report['successful']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--result-root', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.result_root.resolve()) else 1)
