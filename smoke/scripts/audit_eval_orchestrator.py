'''Independently rebuild successful high-eval tables and check recorded smoke PIDs.'''
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = ['motion', 'hypo', 'pattern', 'nav', 'view', 'geo']

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root, report = args.result_root.resolve(), args.report.resolve()
    storage = (REPO / 'smoke/results').resolve()
    if not root.is_relative_to(storage) or not report.is_relative_to(root):
        parser.error('Evidence must remain below the selected smoke/results root')
    checks = []
    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)
    summaries = []
    for label in ['cpu', 'cpu-retry-common', 'cpu-retry-round1', 'cpu-retry-round2']:
        summaries.extend((root / label).glob('cases/*/records_root/*/stage-runs/eval/*/eval-summary.json'))
    adoption = read(root / 'real-adoption/audit.json')
    check('Prior real-output audit completed', adoption['passed'])
    summaries.extend(Path(c['run_dir']) / 'eval-summary.json' for c in adoption['cases'])
    audited = []
    for path in sorted(summaries):
        summary = read(path)
        if summary['status'] != 'succeeded':
            continue
        inv = Path(summary['invocation_dir'])
        result = read(inv / 'results.json')
        matrix = read(inv / 'matrix.json')
        rows = result['rows']
        tag = str(inv)
        def ok(name, passed):
            check(tag + ': ' + name, passed)
        ok('Export hashes', all(digest(meta['path']) == meta['sha256'] for meta in summary['result_files'].values()))
        ok('Summary/results/matrix agree', matrix == result['matrix'] == summary['matrix'] and result['overall'] == summary['overall'])
        ok('Unique complete pair order', [r['pair_id'] for r in rows] == result['pair_order'] and len(rows) == len(set(result['pair_order'])) == summary['pair_count'])
        setting = result['setting']
        axes = result['task_order'] if setting == 'cl' else ['base'] if setting == 'zero_shot' else ['whole'] if setting == 'multitask' else TASKS
        ok('Matrix axes', matrix['row_order'] == axes and matrix['column_order'] == TASKS)
        expected = {(a, b) for a in axes for b in TASKS if setting in ['zero_shot', 'multitask', 'cl'] or (a == b if setting == 'singletask' else a != b)}
        actual = {(r['matrix_row'], r['matrix_column']) for r in rows}
        ok('Exact setting cells', actual == expected and len(rows) == len(expected))
        total, score, invalid = 0, 0.0, 0
        by_cell = {}
        for row in rows:
            output = Path(row['output_dir'])
            scored = [json.loads(line) for line in (output / 'judge.scored.jsonl').read_text().splitlines()]
            raw = [json.loads(line) for line in (output / 'judge.raw.jsonl').read_text().splitlines()]
            value = sum(s['sample_score'] for s in scored)
            bad = sum(s['status'] != 'valid' for s in scored)
            name = row['pair_id']
            ok(name + ' raw/scored count and score', len(raw) == len(scored) == row['total'] and math.isclose(value, row['score_sum'], abs_tol=1e-12) and math.isclose(value / len(scored), row['combined_flat_rawacc'], abs_tol=1e-12))
            ok(name + ' invalid denominator', bad == row['invalid'] and all(s['sample_score'] == 0 for s in scored if s['status'] != 'valid'))
            if setting.startswith('transfer_'):
                expected_task = row['source_task'] if setting == 'transfer_round1' else row['second_round_task']
                ok(name + ' transfer task lineage', row['test_task'] == expected_task and row['matrix_row'] == row['source_task'] and row['matrix_column'] == row['second_round_task'])
            elif setting == 'cl':
                ok(name + ' CL stage lineage', row['cl_stage'] == result['task_order'].index(row['train_task']) + 1 and row['test_task'] == row['matrix_column'])
            total += len(scored)
            score += value
            invalid += bad
            by_cell[row['matrix_row'], row['matrix_column']] = row
        ok('Independent weighted overall', total == summary['overall']['total'] and invalid == summary['overall']['invalid'] and math.isclose(score, summary['overall']['score_sum'], abs_tol=1e-12) and math.isclose(score / total, summary['overall']['combined_flat_rawacc'], abs_tol=1e-12))
        table = list(csv.DictReader((inv / 'results.csv').open()))
        ok('Long CSV pair order and numbers', len(table) == len(rows) and all(c['pair_id'] == r['pair_id'] and int(c['total']) == r['total'] and math.isclose(float(c['score_sum']), r['score_sum'], abs_tol=1e-12) and math.isclose(float(c['combined_flat_rawacc']), r['combined_flat_rawacc'], abs_tol=1e-12) for c, r in zip(table, rows)))
        csv_matrix = list(csv.reader((inv / 'matrix.csv').open()))
        ok('Matrix CSV axes', csv_matrix[0][1:] == TASKS and [r[0] for r in csv_matrix[1:]] == axes)
        for ri, axis in enumerate(axes):
            for ci, task in enumerate(TASKS):
                row = by_cell.get((axis, task))
                if row is None:
                    valid = matrix['values'][ri][ci] is None and matrix['sample_counts'][ri][ci] is None and matrix['pair_ids'][ri][ci] is None and csv_matrix[ri+1][ci+1] == ''
                else:
                    valid = matrix['values'][ri][ci] == row['combined_flat_rawacc'] and matrix['sample_counts'][ri][ci] == row['total'] and matrix['pair_ids'][ri][ci] == row['pair_id'] and float(csv_matrix[ri+1][ci+1]) == row['combined_flat_rawacc']
                ok('Matrix cell ' + axis + '/' + task, valid)
        audited.append({'summary': str(path), 'setting': setting, 'pairs': len(rows), 'total': total})
    pids = set()
    receipts = list(root.rglob('execution.json'))
    for name in ['cpu-high-job.json', 'cpu-slow-worker.json', 'worker-pids.json', 'pid-audit.json']:
        receipts.extend(root.rglob(name))
    for path in receipts:
        try:
            doc = read(path)
        except json.JSONDecodeError:
            continue  # Negative tests deliberately preserve a literal sentinel receipt.
        for key in ['pid', 'launcher_pid', 'worker', 'descendant']:
            value = doc.get(key)
            if type(value) is int and value > 1:
                pids.add(value)
        pids.update(v for v in doc.get('pids', []) if type(v) is int and v > 1)
        for item in doc.get('launches', []):
            if type(item.get('pid')) is int:
                pids.add(item['pid'])
    pids.update(c['pid'] for c in adoption['cases'])
    pids.update(c['pid'] for c in read(root / 'retry-launches.json'))
    pids.update(c['pid'] for c in read(root / 'regression-launches.json'))
    states = []
    for pid in sorted(pids):
        proc = Path('/proc') / str(pid)
        try:
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            cmd = (proc / 'cmdline').read_text().replace(chr(0), ' ')
        except FileNotFoundError:
            state, cmd = 'absent', ''
        states.append({'pid': pid, 'state': state, 'command': cmd})
    check('All recorded processes stopped', all(s['state'] in ['absent', 'Z'] for s in states))
    payload = {'passed': True, 'checks': len(checks), 'details': checks, 'successful_invocations': audited, 'recorded_pids': len(pids), 'process_states': states, 'gpu_work': False}
    with report.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: payload[k] for k in ['passed', 'checks', 'recorded_pids', 'gpu_work']}))
    print(json.dumps({'successful_invocations': len(audited)}))

if __name__ == '__main__':
    main()
