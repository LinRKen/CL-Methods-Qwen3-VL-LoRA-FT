"""Schedule one frozen infer stage over its Adapter/base low-level jobs."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import check_manifest
from .common import fingerprint, new_id, read_json, utc_now, write_json_atomic
from .config import resolve_job_config
from .infer_worker import inspect_predictions, load_samples
from .records import (
    load_experiment, load_job, load_stage_run, record_status, update_status,
)
from .tune_orchestrator import _successful_adapter

if __package__ == "lib":
    from pipeline import build_stage_plan
else:
    from ..pipeline import build_stage_plan

SUPPORTED_SETTINGS = ("zero_shot", "multitask", "singletask", "cl",
                      "transfer_round1", "transfer_round2")


def _context(run_dir):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    run, snapshot = load_stage_run(run_dir)
    if run["stage"] != "infer" or run_dir.name != run["run_id"]:
        raise ValueError("--run-dir must identify a registered infer stage run")
    config = snapshot["config"]
    experiment_config = config["experiment"]
    setting = experiment_config["setting"]
    if setting not in SUPPORTED_SETTINGS:
        raise ValueError("High-level infer does not support setting: " + setting)
    if "infer" not in experiment_config["enabled_stages"]:
        raise ValueError("The experiment must enable the infer stage")
    experiment_id = run["experiment_id"]
    experiment_dir = Path(experiment_config["records_root"]).resolve() / experiment_id
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    if (run_dir != experiment_dir / "stage-runs" / "infer" / run["run_id"]
            or experiment["experiment_id"] != experiment_id
            or experiment["status"] != "active"
            or run["status"] not in ("planned", "running")
            or snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or experiment_config != experiment_snapshot["config"]["experiment"]
            or run["config_fingerprint"] != snapshot["config_fingerprint"]
            or {"stage": "infer", "run_id": run["run_id"],
                "path": run_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]):
        raise ValueError("Infer run is not an active child of its frozen experiment")
    if run["plan"] != "plan.json":
        raise ValueError("Infer orchestration requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    expected_plan = build_stage_plan(snapshot, "infer")
    if plan != expected_plan or run["plan_fingerprint"] != fingerprint(plan):
        raise ValueError("Frozen infer plan differs from the plan derived from its snapshot")

    planned = plan["jobs"]
    registered = run["jobs"]
    expected_ids = [item["job_id"] for item in planned]
    if ([item.get("job_id") for item in registered] != expected_ids
            or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError("Registered infer jobs do not match the frozen plan and order")
    jobs = {}
    for registration, item in zip(registered, planned):
        job_id = item["job_id"]
        job_dir = run_dir / "jobs" / job_id
        job, job_snapshot = load_job(job_dir)
        resolved = resolve_job_config(snapshot, "infer", job_id,
                                      job_snapshot["layers"].get("job"),
                                      job_snapshot["layers"].get("job_cli"))
        if (registration.get("path") != "jobs/" + job_id
                or job["job_id"] != job_id or job["stage"] != "infer"
                or job["experiment_id"] != experiment_id or job["run_id"] != run["run_id"]
                or job_dir != run_dir / "jobs" / job["job_id"]
                or job["config_fingerprint"] != resolved["config_fingerprint"]
                or job_snapshot["config"]["experiment"] != experiment_config
                or job_snapshot["layers"].get("parent_snapshot") != snapshot["snapshot_fingerprint"]
                or job["task"] is not None or job["tasks"] != item.get("tasks")
                or job["depends_on"] != item.get("depends_on", [])
                or job["requires"] != item.get("requires", [])
                or job["input_specs"] != item.get("input_specs", [])
                or job["lineage"] != item.get("lineage", {})
                or job["output_spec"] != item.get("output")
                or Path(job["output_dir"]) != (Path(config["experiment"]["infer_root"]).resolve()
                                                / experiment_id / run["run_id"] / job_id)):
            raise ValueError("Infer job record differs from its frozen plan: " + job_id)
        if job["status"] not in ("planned", "succeeded"):
            raise ValueError("Infer job cannot be resumed from status {}: {}".format(job["status"], job_id))
        if job["status"] == "planned" and (job["inputs"] or job["artifacts"]
                or (job_dir / "execution.json").exists() or Path(job["output_dir"]).exists()
                or Path(job["output_dir"]).is_symlink()):
            raise ValueError("A planned infer job already has bound inputs or execution payload: " + job_id)
        jobs[job_id] = {"directory": job_dir, "record": job, "snapshot": job_snapshot,
                        "planned": item, "adapter": None, "initial_status": job["status"],
                        "resources": _resources(job_snapshot["config"], job_id)}

    if setting != "zero_shot":
        registered_tune = [item for item in experiment["stage_runs"] if item["stage"] == "tune"]
        if "tune" not in experiment_config["enabled_stages"] or len(registered_tune) != 1:
            raise ValueError("This setting requires its registered tune stage to finish before inference")
        registration = registered_tune[0]
        tune_dir = experiment_dir / "stage-runs" / "tune" / registration["run_id"]
        if registration["path"] != tune_dir.relative_to(experiment_dir).as_posix():
            raise ValueError("Registered tune stage path differs from its identity")
        tune_run, tune_snapshot = load_stage_run(tune_dir)
        if (tune_run["stage"] != "tune" or tune_run["status"] != "succeeded"
                or tune_run["run_id"] != registration["run_id"]
                or tune_run["experiment_id"] != experiment_id
                or tune_snapshot["config"]["experiment"] != experiment_config
                or tune_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
                or {"stage": "tune", "run_id": tune_run["run_id"],
                    "path": tune_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]):
            raise ValueError("The exact tune stage for this experiment has not succeeded")
        _bind_adapter_sources(jobs, experiment_id, tune_dir, tune_run)

    for job_id, info in jobs.items():
        if info["initial_status"] == "succeeded":
            _audit_succeeded_job(info, config["experiment"]["dataset_root"])
    return run_dir, run, snapshot, experiment_dir, setting, jobs


def _resources(config, job_id):
    infer = config["infer"]
    groups = infer["run"]["gpu_groups"]
    tensor_parallel_size = infer["params"]["tensor_parallel_size"]
    resources = []
    for index, group in enumerate(groups):
        if (not group or len(group) != tensor_parallel_size
                or any(type(gpu) is not int or gpu not in (1, 2, 3) for gpu in group)
                or len(set(group)) != len(group)):
            raise ValueError("Infer job {} must use unique physical GPU groups 1-3 matching tensor_parallel_size".format(job_id))
        resources.append({"group_index": index, "physical_gpu_ids": list(group)})
    if not resources:
        raise ValueError("Infer job has no configured physical GPU group: " + job_id)
    return resources


def _bind_adapter_sources(jobs, experiment_id, tune_dir, tune_run):
    tune_jobs = {item["job_id"]: tune_dir / "jobs" / item["job_id"] for item in tune_run["jobs"]}
    for job_id, info in jobs.items():
        job = info["record"]
        references = job["requires"]
        if len(references) != 1:
            raise ValueError("Every trained setting infer job must require exactly one adapter: " + job_id)
        reference = references[0]
        source_id = job["lineage"].get("adapter_ref")
        if (reference.get("stage") != "tune" or reference.get("artifact_kind") != "adapter"
                or reference.get("job_id") != source_id or reference.get("slot") != job["output_spec"]["slot"]
                or source_id not in tune_jobs):
            raise ValueError("Infer adapter reference differs from the frozen setting lineage: " + job_id)
        source_dir = tune_jobs[source_id]
        source_job, _ = load_job(source_dir)
        if source_job["run_id"] != tune_run["run_id"] or source_job["experiment_id"] != experiment_id:
            raise ValueError("Adapter source is outside this experiment's tune run: " + job_id)
        adapter = _successful_adapter(source_dir, source_job)
        if adapter["slot"] != reference["slot"]:
            raise ValueError("Adapter slot differs from the infer requirement: " + job_id)
        info["adapter"] = adapter


def _artifact_map(job_dir, job):
    result = {}
    for item in job["artifacts"]:
        if item["role"] in result:
            raise ValueError("Duplicate registered artifact role: " + item["role"])
        if item["path"] != "artifacts/" + item["role"] + ".json":
            raise ValueError("Artifact manifest path differs from its role")
        manifest = check_manifest(read_json(job_dir / item["path"]), verify_contents=True)
        if (manifest["kind"] != item["kind"] or manifest["fingerprint"] != item["fingerprint"]
                or manifest["producer"] != {key: job[key] for key in
                                               ("experiment_id", "stage", "run_id", "job_id")}):
            raise ValueError("Registered infer artifact does not match its job record")
        result[item["role"]] = manifest
    return result


def _audit_succeeded_job(info, dataset_root):
    job_dir, job = info["directory"], info["record"]
    tasks = job["tasks"]
    expected_inputs = {spec["role"] for spec in job["input_specs"]}
    if info["adapter"] is not None:
        expected_inputs.add("adapter")
    if set(job["inputs"]) != expected_inputs:
        raise ValueError("Succeeded infer job input roles are incomplete: " + job["job_id"])
    for manifest in job["inputs"].values():
        check_manifest(manifest, verify_contents=True)
        if Path(manifest["requested_path"]).resolve(strict=True) != Path(manifest["path"]):
            raise ValueError("Succeeded infer input path was redirected: " + job["job_id"])
    for spec in job["input_specs"]:
        manifest = job["inputs"][spec["role"]]
        if (manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != str(Path(spec["path"]).absolute())
                or manifest["external_source"] != spec["source"]
                or manifest["metadata"] != spec["metadata"]):
            raise ValueError("Succeeded infer input differs from its frozen specification: " + spec["role"])
    if info["adapter"] is not None:
        adapter = job["inputs"].get("adapter")
        if (adapter is None or adapter["fingerprint"] != info["adapter"]["fingerprint"]
                or adapter["producer"] != info["adapter"]["producer"]):
            raise ValueError("Succeeded infer job is bound to a different adapter: " + job["job_id"])
    execution_path = job_dir / "execution.json"
    execution = read_json(execution_path)
    groups = info["resources"]
    if (execution.get("status") != "succeeded" or execution.get("cleanup_completed") is not True
            or execution.get("job_id") != job["job_id"]
            or execution.get("config_fingerprint") != job["config_fingerprint"]
            or execution.get("tasks") != tasks
            or execution.get("output_dir") != job["output_dir"]
            or type(execution.get("gpu_group_index")) is not int
            or not 0 <= execution["gpu_group_index"] < len(groups)
            or execution.get("physical_gpu_ids") != groups[execution["gpu_group_index"]]["physical_gpu_ids"]):
        raise ValueError("Successful infer execution receipt is missing or inconsistent: " + job["job_id"])
    request_path = job_dir / "inference.request.json"
    request = read_json(request_path)
    payload = {key: value for key, value in request.items() if key != "request_fingerprint"}
    if (request.get("schema_version") != 1 or request.get("request_fingerprint") != fingerprint(payload)
            or request.get("producer") != {key: job[key] for key in
                                            ("experiment_id", "stage", "run_id", "job_id")}
            or request.get("config_fingerprint") != job["config_fingerprint"]
            or request.get("lineage") != job["lineage"]
            or request.get("repo_root") != info["snapshot"]["config"]["experiment"]["repo_root"]
            or request.get("media_root") != str(Path(dataset_root).resolve(strict=True))
            or request.get("output_dir") != job["output_dir"]
            or request.get("base_model") != job["inputs"]["base-model"]["path"]
            or request.get("processor_path") != job["inputs"]["processor"]["path"]
            or request.get("adapter_path") != (info["adapter"]["path"] if info["adapter"] else None)
            or request.get("tasks") is None or [item.get("task") for item in request["tasks"]] != tasks
            or request.get("physical_gpu_ids") != execution["physical_gpu_ids"]
            or request.get("input_fingerprints") != {role: value["fingerprint"]
                                                       for role, value in job["inputs"].items()}
            or request.get("params") != info["snapshot"]["config"]["infer"]["params"]
            or request.get("adapter_fingerprint") != (info["adapter"]["fingerprint"]
                                                       if info["adapter"] else None)):
        raise ValueError("Successful inference request does not match its frozen job: " + job["job_id"])
    outputs = _artifact_map(job_dir, job)
    expected_roles = {"prediction-" + task for task in tasks} | {"inference-summary"}
    if set(outputs) != expected_roles:
        raise ValueError("Successful infer job has an incomplete prediction artifact set: " + job["job_id"])
    inspected = {}
    for task_spec in request["tasks"]:
        task = task_spec["task"]
        annotation = job["inputs"].get("annotation-{}-test".format(task))
        if (annotation is None or task_spec.get("annotation_path") != annotation["path"]
                or task_spec.get("annotation_sha256") != annotation["identity"].get("sha256")
                or task_spec.get("dataset") != "cosine_{}_test".format(task)
                or task_spec.get("output_path") != str(Path(job["output_dir"]) / "predictions" / (task + ".jsonl"))):
            raise ValueError("Inference request test input differs from its frozen annotation: " + task)
        samples = load_samples(annotation["path"], dataset_root)
        result = inspect_predictions(task_spec, samples, request["params"])
        manifest = outputs["prediction-" + task]
        metadata = manifest["metadata"]
        if (manifest["kind"] != "prediction"
                or manifest["requested_path"] != task_spec["output_path"]
                or manifest["path"] != str(Path(task_spec["output_path"]).resolve(strict=True))
                or metadata.get("slot") != job["output_spec"]["slot"]
                or metadata.get("task") != task or metadata.get("split") != "test"
                or metadata.get("dataset") != task_spec["dataset"]
                or metadata.get("sample_count") != result["sample_count"]
                or metadata.get("empty_predictions") != result["empty_predictions"]
                or metadata.get("request_fingerprint") != request["request_fingerprint"]
                or metadata.get("config_fingerprint") != job["config_fingerprint"]
                or metadata.get("lineage") != job["lineage"]
                or metadata.get("input_fingerprints") != request["input_fingerprints"]
                or metadata.get("base_model_fingerprint") != job["inputs"]["base-model"]["fingerprint"]
                or metadata.get("processor_fingerprint") != job["inputs"]["processor"]["fingerprint"]
                or metadata.get("adapter_fingerprint") != request["adapter_fingerprint"]
                or metadata.get("params") != request["params"]
                or metadata.get("model_loads_per_job") != 1
                or metadata.get("annotation_fingerprint") != annotation["fingerprint"]):
            raise ValueError("Prediction artifact failed its complete test-set audit: " + task)
        inspected[task] = result
    summary_manifest = outputs["inference-summary"]
    summary_path = Path(job["output_dir"]) / "inference_result.json"
    summary = read_json(summary_path)
    summary_meta = summary_manifest["metadata"]
    if (summary_manifest["kind"] != "summary"
            or summary_manifest["requested_path"] != str(summary_path)
            or summary_manifest["path"] != str(summary_path.resolve(strict=True))
            or summary_meta.get("slot") != job["output_spec"]["slot"]
            or summary_meta.get("tasks") != tasks
            or summary_meta.get("config_fingerprint") != job["config_fingerprint"]
            or summary_meta.get("request_fingerprint") != request["request_fingerprint"]
            or summary.get("schema_version") != 1 or summary.get("status") != "succeeded"
            or summary.get("producer") != request["producer"]
            or summary.get("config_fingerprint") != job["config_fingerprint"]
            or summary.get("request_fingerprint") != request["request_fingerprint"]
            or summary.get("input_fingerprints") != request["input_fingerprints"]
            or summary.get("params") != request["params"]
            or summary.get("physical_gpu_ids") != request["physical_gpu_ids"]
            or summary.get("model_loads") != 1 or summary.get("engine_shutdown") is not True
            or summary.get("task_order") != tasks or not isinstance(summary.get("tasks"), list)
            or len(summary["tasks"]) != len(tasks)):
        raise ValueError("Inference summary is incomplete or inconsistent: " + job["job_id"])
    for actual, task in zip(summary["tasks"], tasks):
        if any(actual.get(key) != value for key, value in inspected[task].items()):
            raise ValueError("Inference summary differs from prediction payload: " + task)


@contextmanager
def _run_lock(run_dir):
    path = run_dir / ".infer-orchestrator.lock"
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another high-level infer run is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _signals():
    state = {"number": None}

    def interrupt(signum, frame):
        state["number"] = signum

    previous = {number: signal.signal(number, interrupt)
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _mark(job_dir, status, message):
    current = record_status(job_dir)["status"]
    if current in ("planned", "running"):
        update_status(job_dir, status, message)


def _terminate(active):
    for item in active.values():
        if item["process"].poll() is None:
            try:
                item["process"].send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
    for item in active.values():
        try:
            item["process"].wait()
        finally:
            item["log"].close()


def _launch(job_id, info, resource, repo_root, low_level, invocation_dir, receipt, active):
    log_path = invocation_dir / (job_id + ".log")
    job, config = info["record"], info["snapshot"]["config"]
    manifest_path = info["adapter"]["manifest"] if info["adapter"] else None
    command = [config["infer"]["run"]["python"], str(low_level),
               "--job-dir", str(info["directory"]),
               "--gpu-group-index", str(resource["group_index"])]
    if manifest_path:
        command.extend(["--input", "adapter=" + manifest_path])
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, resource["physical_gpu_ids"])),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    launch = {"job_id": job_id, "command": command, "assignment": resource,
              "log": str(log_path), "started_at": utc_now(), "pid": None, "status": "launching"}
    info.update(log_path=str(log_path), assignment=resource)
    receipt["launches"].append(launch)
    write_json_atomic(invocation_dir / "execution.json", receipt)
    log = log_path.open("xb")
    try:
        print("Starting {} on physical GPUs {}; log: {}".format(
            job_id, resource["physical_gpu_ids"], log_path), flush=True)
        process = subprocess.Popen(command, cwd=str(repo_root), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException as error:
        log.close()
        launch.update(status="launch_failed", error=str(error), finished_at=utc_now())
        raise
    active[job_id] = {"process": process, "log": log, "resource": resource, "launch": launch}
    launch.update(pid=process.pid, status="running")


def _collect(active, jobs, failures, *, stop_message=None):
    completed, unsafe = 0, []
    for job_id, item in list(active.items()):
        returncode = item["process"].poll()
        if returncode is None:
            continue
        item["log"].close()
        info = jobs[job_id]
        record, snapshot = load_job(info["directory"])
        info.update(record=record, returncode=returncode)
        audit_error = None
        if returncode == 0 and record["status"] == "succeeded":
            try:
                if snapshot["snapshot_fingerprint"] != info["snapshot"]["snapshot_fingerprint"]:
                    raise ValueError("Frozen job configuration changed during inference")
                _audit_succeeded_job(info, info["snapshot"]["config"]["experiment"]["dataset_root"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit_error = str(error)
        execution_path = info["directory"] / "execution.json"
        try:
            execution = read_json(execution_path) if execution_path.is_file() else {}
            cleanup_error = execution.get("cleanup_error")
        except (OSError, ValueError, KeyError, TypeError) as error:
            execution, cleanup_error = {}, "Cannot read low-level execution receipt: {}".format(error)
        incomplete_shutdown = (record["status"] == "running"
                               or execution.get("cleanup_completed") is not True
                               or bool(cleanup_error))
        if (returncode != 0 or record["status"] != "succeeded" or audit_error
                or cleanup_error or incomplete_shutdown):
            message = "Low-level infer job exited {} with status {}".format(returncode, record["status"])
            if audit_error:
                message += "; prediction audit failed: " + audit_error
            if cleanup_error:
                message += "; worker cleanup failed: " + str(cleanup_error)
            elif incomplete_shutdown:
                message += "; worker cleanup was not confirmed"
            if stop_message:
                message += "; " + stop_message
            _mark(info["directory"], "cancelled" if stop_message else "failed", message)
            failures.append({"job_id": job_id, "returncode": returncode, "message": message})
        status = record_status(info["directory"])["status"]
        item["launch"].update(returncode=returncode, finished_at=utc_now(), status=status)
        del active[job_id]
        completed += 1
        if incomplete_shutdown:
            unsafe.append(job_id)
        print("Finished {}: {} (exit {})".format(job_id, status, returncode), flush=True)
    return completed, unsafe


def _cancel(pending, active, jobs, failures, message):
    _terminate(active)
    _collect(active, jobs, failures, stop_message=message)
    for job_id in list(pending):
        _mark(jobs[job_id]["directory"], "cancelled", message + "; job was not launched")
        pending.remove(job_id)


def _summary(run_dir, invocation_dir, run, setting, jobs, started_at, status, details):
    entries = []
    for job_id, info in jobs.items():
        record = record_status(info["directory"])
        entries.append({"job_id": job_id, "tasks": record["tasks"], "status": record["status"],
                        "config_fingerprint": record["config_fingerprint"],
                        "output_dir": record["output_dir"], "initial_status": info["initial_status"],
                        "returncode": info.get("returncode"), "log": info.get("log_path"),
                        "assignment": info.get("assignment"), "adapter": info["adapter"],
                        "artifacts": [{**artifact, "manifest": str(info["directory"] / artifact["path"])}
                                      for artifact in record["artifacts"]]})
    result = {"schema_version": 1, "kind": "infer_run_summary",
              "experiment_id": run["experiment_id"], "run_id": run["run_id"],
              "setting": setting, "status": status, "job_order": list(jobs),
              "task_order": {job_id: info["record"]["tasks"] for job_id, info in jobs.items()},
              "config_fingerprint": run["config_fingerprint"], "invocation_dir": str(invocation_dir),
              "started_at": started_at, "finished_at": utc_now(), "jobs": entries}
    if details:
        result["details"] = details
    write_json_atomic(invocation_dir / "summary.json", result)
    write_json_atomic(run_dir / "infer-summary.json", result)
    return result


def execute_infer_run(run_dir):
    """Schedule each frozen Adapter/base job once, preserving one vLLM load per job."""
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    with _run_lock(run_dir), _signals() as signals:
        run_dir, run, snapshot, experiment_dir, setting, jobs = _context(run_dir)
        config = snapshot["config"]
        infer_run = config["infer"]["run"]
        capacity, fail_fast = infer_run["max_parallel_jobs"], infer_run["fail_fast"]
        repo_root = Path(config["experiment"]["repo_root"])
        low_level = repo_root / "scripts" / "experiments" / "run_infer_job.py"
        if not low_level.is_file():
            raise ValueError("Low-level infer entry point is missing: {}".format(low_level))
        started_at = utc_now()
        invocation_dir = run_dir / "orchestrations" / new_id("infer")
        invocation_dir.mkdir(parents=True, exist_ok=False)
        receipt = {"schema_version": 1, "run_id": run["run_id"],
                   "experiment_id": run["experiment_id"],
                   "config_fingerprint": run["config_fingerprint"], "started_at": started_at,
                   "pid": os.getpid(), "status": "running", "max_parallel_jobs": capacity,
                   "fail_fast": fail_fast, "setting": setting, "job_order": list(jobs), "launches": []}
        write_json_atomic(invocation_dir / "execution.json", receipt)
        pending = [job_id for job_id, info in jobs.items() if info["initial_status"] == "planned"]
        active, failures = {}, []
        interrupted = False
        try:
            if signals["number"] is not None:
                raise InterruptedError("Infer orchestrator interrupted during preparation")
            if run["status"] == "planned":
                update_status(run_dir, "running", "High-level {} infer run started".format(setting))
            while pending or active:
                if signals["number"] is not None:
                    raise InterruptedError("Infer orchestrator received signal {}".format(signals["number"]))
                completed, unsafe = _collect(active, jobs, failures)
                if completed:
                    write_json_atomic(invocation_dir / "execution.json", receipt)
                if unsafe:
                    _cancel(pending, active, jobs, failures,
                            "Low-level worker shutdown is unconfirmed for {}; review cleanup".format(unsafe))
                    break
                if fail_fast and failures:
                    _cancel(pending, active, jobs, failures, "Stopped after another infer job failed")
                    break
                if pending:
                    busy_gpus = {gpu for item in active.values()
                                 for gpu in item["resource"]["physical_gpu_ids"]}
                    for job_id in list(pending):
                        if signals["number"] is not None:
                            raise InterruptedError("Infer orchestrator interrupted before launch")
                        if len(active) >= capacity:
                            break
                        info = jobs[job_id]
                        resource = next((item for item in info["resources"]
                                         if not busy_gpus.intersection(item["physical_gpu_ids"])), None)
                        if resource is None:
                            continue
                        try:
                            _launch(job_id, info, resource, repo_root, low_level,
                                    invocation_dir, receipt, active)
                        except (OSError, ValueError, KeyError) as error:
                            pending.remove(job_id)
                            _mark(info["directory"], "failed",
                                  "Cannot launch low-level infer executor: {}".format(error))
                            failures.append({"job_id": job_id, "message": str(error)})
                            write_json_atomic(invocation_dir / "execution.json", receipt)
                            if fail_fast:
                                break
                            continue
                        pending.remove(job_id)
                        write_json_atomic(invocation_dir / "execution.json", receipt)
                        busy_gpus.update(resource["physical_gpu_ids"])
                        if len(active) >= capacity:
                            break
                if active:
                    time.sleep(0.1)
                elif pending and not (fail_fast and failures):
                    raise ValueError("Pending infer jobs have no available GPU assignment: {}".format(pending))
            if signals["number"] is not None:
                raise InterruptedError("Infer orchestrator interrupted during execution")
        except (KeyboardInterrupt, InterruptedError):
            interrupted = True
            _cancel(pending, active, jobs, failures, "High-level infer run interrupted")
        except BaseException as error:
            _cancel(pending, active, jobs, failures, "Scheduler stopped: {}".format(error))
            failures.append({"message": "{}: {}".format(type(error).__name__, error)})

        child_statuses = [record_status(info["directory"])["status"] for info in jobs.values()]
        final_status = ("cancelled" if interrupted else
                        "succeeded" if not failures and child_statuses
                        and all(item == "succeeded" for item in child_statuses) else "failed")
        summary = _summary(run_dir, invocation_dir, run, setting, jobs,
                           started_at, final_status, failures)
        update_status(run_dir, final_status,
                      "All {} infer jobs completed".format(len(jobs)) if final_status == "succeeded"
                      else "High-level infer run {}".format(final_status))
        receipt.update(status=final_status, finished_at=utc_now(),
                       summary=str(invocation_dir / "summary.json"))
        write_json_atomic(invocation_dir / "execution.json", receipt)
        experiment, _ = load_experiment(experiment_dir)
        if (final_status == "succeeded" and experiment["status"] == "active"
                and experiment["stage_runs"]
                and all(record_status(experiment_dir / item["path"])["status"] == "succeeded"
                        for item in experiment["stage_runs"])):
            update_status(experiment_dir, "succeeded", "All registered stage runs completed")
        return summary

