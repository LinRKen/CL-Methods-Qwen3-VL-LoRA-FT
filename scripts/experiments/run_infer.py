"""Run one frozen infer stage over all its low-level Adapter/base jobs."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    from lib.infer_orchestrator import execute_infer_run
else:
    from .lib.infer_orchestrator import execute_infer_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Absolute existing infer stage-run directory; use frozen configuration")
    arguments = parser.parse_args(argv)
    try:
        if not arguments.run_dir.is_absolute():
            raise ValueError("--run-dir must be an absolute path")
        result = execute_infer_run(arguments.run_dir)
    except (KeyboardInterrupt, InterruptedError) as error:
        print(str(error) or "Infer orchestration interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return {"succeeded": 0, "failed": 1, "cancelled": 130}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

