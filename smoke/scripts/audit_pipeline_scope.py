"""Capture/verify non-smoke files and Git state without changing Git or using GPUs.

Capture before a test, then verify against that baseline after the test. All
outputs must stay below smoke/results. Baseline omissions are never backfilled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True,
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})


def state():
    files = {}
    for directory, dirs, names in os.walk(REPO, followlinks=False):
        if Path(directory) == REPO:
            dirs[:] = [d for d in dirs if d not in ('.git', 'smoke')]
        for name in names:
            path = Path(directory) / name
            files[str(path.relative_to(REPO))] = {
                'mode': path.lstat().st_mode,
                'sha256': digest(path) if path.is_file() else None,
                'link': os.readlink(path) if path.is_symlink() else None,
            }
    index = Path(git('rev-parse', '--git-path', 'index').strip())
    if not index.is_absolute():
        index = REPO / index
    return {
        'head': git('rev-parse', 'HEAD'),
        'branch': git('branch', '--show-current'),
        'refs': git('show-ref'),
        'status': git('status', '--short', '--untracked-files=all'),
        'diff': git('diff', '--binary', '--', '.', ':(exclude)smoke'),
        'staged_diff': git('diff', '--cached', '--binary'),
        'index_sha256': digest(index),
        'files': files,
        'smoke_links': {name: os.readlink(REPO/'smoke'/name) for name in ('data', 'results')},
    }


def non_smoke_status(text):
    return [line for line in text.splitlines() if not line[3:].startswith('smoke/')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'verify'))
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    storage = (REPO/'smoke/results').resolve(strict=True)
    for path in (args.baseline, args.report):
        if path is not None and not path.resolve().is_relative_to(storage):
            parser.error('Evidence files must remain below smoke/results')
    current = state()
    if args.mode == 'capture':
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with args.baseline.open('x') as handle:
            json.dump(current, handle, indent=2, ensure_ascii=False)
        print(json.dumps({'baseline': str(args.baseline), 'files': len(current['files'])}))
        return 0
    if args.report is None:
        parser.error('verify requires --report')
    baseline = json.loads(args.baseline.read_text())
    checks = {name: current[name] == baseline[name] for name in (
        'head', 'branch', 'refs', 'diff', 'staged_diff', 'index_sha256', 'smoke_links')}
    checks['original_non_smoke_status'] = non_smoke_status(current['status']) == non_smoke_status(baseline['status'])
    changed = sorted(name for name in set(current['files']) | set(baseline['files'])
                     if current['files'].get(name) != baseline['files'].get(name))
    checks['non_smoke_files_hashes_permissions_links'] = not changed
    process = subprocess.run(['git','diff','--check'], cwd=REPO, text=True, capture_output=True,
                             env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    checks['diff_check'] = process.returncode == 0
    changes = git('status','--short','--untracked-files=all','--','smoke')
    report = {'successful': all(checks.values()), 'checks': checks,
              'baseline_non_smoke_files': len(baseline['files']), 'changed_non_smoke_files': changed,
              'smoke_changes': changes.splitlines(), 'diff_check_stdout': process.stdout,
              'diff_check_stderr': process.stderr}
    with args.report.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
