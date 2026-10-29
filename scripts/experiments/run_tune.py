"""High-level tune entry point using frozen job records."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    from lib.config import dotted_overrides, load_config
    from lib.orchestration_logging import record_preparation_failure
    from lib.records import load_experiment
    from lib.tune_orchestrator import execute_tune_run, require_supported_setting
    from pipeline import prepare_records
else:
    from .lib.config import dotted_overrides, load_config
    from .lib.orchestration_logging import record_preparation_failure
    from .lib.records import load_experiment
    from .lib.tune_orchestrator import execute_tune_run, require_supported_setting
    from .pipeline import prepare_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Multitask, Singletask, CL or transfer setting YAML for a new experiment")
    source.add_argument("--run-dir", type=Path, help="Absolute existing tune stage-run directory; use frozen configuration")
    parser.add_argument("--default", dest="default_path", type=Path, default=None,
                        help="Default YAML for --config; omitted uses configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted YAML override for --config; repeat for multiple values")
    parser.add_argument("--experiment-id", default=None, help="Optional ID for a new experiment")
    args = parser.parse_args(argv)
    if args.run_dir is not None and (args.default_path is not None or args.set or args.experiment_id is not None):
        parser.error("--run-dir cannot be combined with --default, --set or --experiment-id")
    snapshot = None
    execution_started = False
    try:
        if args.config is not None:
            overrides = dotted_overrides(args.set) if args.set else {}
            snapshot = load_config(args.config, default_path=args.default_path, cli_overrides=overrides)
            require_supported_setting(snapshot)
            experiment_dir = prepare_records(snapshot, experiment_id=args.experiment_id)
            experiment, _ = load_experiment(experiment_dir)
            tune = next(item for item in experiment["stage_runs"] if item["stage"] == "tune")
            run_dir = experiment_dir / tune["path"]
            print("Experiment records: {}".format(experiment_dir), flush=True)
        else:
            if not args.run_dir.is_absolute():
                raise ValueError("--run-dir must be an absolute path")
            run_dir = args.run_dir
        print("Tune stage run: {}".format(run_dir), flush=True)
        execution_started = True
        result = execute_tune_run(run_dir)
    except (KeyboardInterrupt, InterruptedError) as error:
        if snapshot is not None and not execution_started:
            record_preparation_failure("tune", snapshot, error)
        print(str(error) or "Tune orchestration interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        if snapshot is not None and not execution_started:
            record_preparation_failure("tune", snapshot, error)
        print("{}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return {"succeeded": 0, "failed": 1, "cancelled": 130}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
