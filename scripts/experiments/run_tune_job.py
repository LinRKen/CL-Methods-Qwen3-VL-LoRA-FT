"""Synchronous low-level entry point for one previously prepared tune job."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    from lib.common import read_json, require_id
    from lib.tune_executor import execute_tune_job
else:
    from .lib.common import read_json, require_id
    from .lib.tune_executor import execute_tune_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Absolute directory containing the frozen job.json")
    parser.add_argument("--gpu-group-index", type=int, default=0,
                        help="Zero-based index in this job's frozen tune.run.gpu_groups")
    parser.add_argument("--input", action="append", default=[], metavar="ROLE=MANIFEST_JSON",
                        help="Bind an explicit input manifest; repeat for different roles")
    parser.add_argument("--resume-training-state", action=argparse.BooleanOptionalAction, default=None,
                        help="Restore one explicit same-task checkpoint; never find a latest checkpoint")
    arguments = parser.parse_args(argv)
    try:
        job_dir = Path(arguments.job_dir)
        if not job_dir.is_absolute():
            raise ValueError("--job-dir must be an absolute path")
        inputs = {}
        for item in arguments.input:
            role, separator, filename = item.partition("=")
            require_id(role)
            if not separator or not Path(filename).is_absolute() or role in inputs:
                raise ValueError("--input requires a unique ROLE and an absolute manifest JSON file")
            inputs[role] = read_json(filename)
        result = execute_tune_job(job_dir, gpu_group_index=arguments.gpu_group_index,
                                  input_manifests=inputs,
                                  resume_training_state=arguments.resume_training_state)
    except (KeyboardInterrupt, InterruptedError) as error:
        print(str(error) or "Training execution interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
