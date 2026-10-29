"""Persistent logs for high-level execution and preflight failures."""

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback

from .common import new_id, utc_now, write_json_atomic
from .records import load_stage_run


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.stream.write(value)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _logs_root(snapshot):
    experiment = snapshot["config"]["experiment"]
    return Path(experiment.get("logs_root", experiment["records_root"]))


def _new_invocation(stage, snapshot, *, run=None, run_dir=None):
    root = _logs_root(snapshot)
    if run is None:
        parent = root / "_preparation" / stage
    else:
        parent = root / run["experiment_id"] / "stage-runs" / stage / run["run_id"] / "orchestrations"
    directory = parent / new_id(stage)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "stage": stage, "pid": os.getpid(),
               "status": "running", "started_at": utc_now(),
               "config_fingerprint": snapshot["config_fingerprint"],
               "invocation_dir": str(directory),
               "orchestrator_log": str(directory / "orchestrator.log")}
    if run is not None:
        receipt.update(experiment_id=run["experiment_id"], run_id=run["run_id"],
                       run_dir=str(run_dir))
    write_json_atomic(directory / "invocation.json", receipt)
    return directory, receipt


def _error_details(error):
    return {"error_type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}


def _finish(directory, receipt, status, *, failure=None):
    receipt.update(status=status, finished_at=utc_now())
    if failure is not None:
        failure = {**receipt, **failure}
        write_json_atomic(directory / "failure.json", failure)
        receipt["failure"] = str(directory / "failure.json")
    write_json_atomic(directory / "invocation.json", receipt)


def run_logged(stage, run_dir, execute):
    """Create logs before business preflight, without changing frozen resources."""
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    run, snapshot = load_stage_run(run_dir)
    if run["stage"] != stage:
        raise ValueError("--run-dir must identify a registered {} stage run".format(stage))
    directory, receipt = _new_invocation(stage, snapshot, run=run, run_dir=run_dir)
    with (directory / "orchestrator.log").open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print("Orchestration logs: {}".format(directory), flush=True)
            try:
                result = execute(run_dir, directory)
            except BaseException as error:
                status = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
                details = _error_details(error)
                try:
                    _finish(directory, receipt, status, failure=details)
                except Exception as log_error:
                    print("Cannot save orchestration failure: {}".format(log_error), file=sys.stderr)
                print(details["traceback"], file=sys.stderr, flush=True)
                raise
            status = result["status"]
            failure = None if status == "succeeded" else {
                "message": "High-level {} run {}".format(stage, status),
                "summary": str(directory / "summary.json"),
                "details": result.get("details", []),
            }
            _finish(directory, receipt, status, failure=failure)
            print("High-level {} run {}".format(stage, status), flush=True)
            if failure is not None:
                print(json.dumps(failure, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)
            return result


def record_preparation_failure(stage, snapshot, error):
    """Persist a failed --config preparation once a valid config is available."""
    try:
        directory, receipt = _new_invocation(stage, snapshot)
        status = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
        details = _error_details(error)
        with (directory / "orchestrator.log").open("x", encoding="utf-8") as log:
            log.write(details["traceback"])
        _finish(directory, receipt, status, failure=details)
    except Exception as log_error:
        print("Cannot save preparation failure: {}".format(log_error), file=sys.stderr)
        return None
    print("Preparation failure logs: {}".format(directory), file=sys.stderr, flush=True)
    return directory
