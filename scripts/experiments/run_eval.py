#!/usr/bin/env python3
"""Run one frozen experiment evaluation stage and summarize its results."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    from lib.eval_orchestrator import execute_eval_run
else:
    from .lib.eval_orchestrator import execute_eval_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Absolute existing eval stage-run directory; use frozen configuration")
    args = parser.parse_args(argv)
    try:
        if not args.run_dir.is_absolute():
            raise ValueError("--run-dir must be an absolute path")
        summary = execute_eval_run(args.run_dir)
    except (KeyboardInterrupt, InterruptedError):
        print("Evaluation orchestration interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print("Evaluation orchestration failed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return {"succeeded": 0, "failed": 1, "cancelled": 130}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
