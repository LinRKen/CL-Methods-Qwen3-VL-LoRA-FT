"""Run one frozen tune stage over its low-level jobs."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from .artifacts import check_manifest, make_manifest
from .common import TASKS, read_json, sha256_file, utc_now, write_json_atomic
from .config import resolve_job_config
from .orchestration_logging import run_logged
from .records import (
    activate_recovery_stage_run, create_job, create_recovery_stage_run, import_recovery_adapter,
    bind_inputs, load_experiment, load_job, load_stage_run, record_status, set_recovery_source, update_status,
)
from .tune_executor import (CONTINUE, FRESH, RESTORE, _checkpoint_state, _context as _tune_job_context)

SUPPORTED_SETTINGS = ("multitask", "singletask", "cl", "transfer_round1", "transfer_round2")


def require_supported_setting(snapshot):
    experiment = snapshot["config"]["experiment"]
    if experiment["setting"] not in SUPPORTED_SETTINGS:
        raise ValueError("High-level tune does not support setting: " + experiment["setting"])
    if "tune" not in experiment["enabled_stages"]:
        raise ValueError("The experiment must enable the tune stage")


def _context(run_dir, *, recovery_source=False):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    run, snapshot = load_stage_run(run_dir)
    if run["stage"] != "tune" or run_dir.name != run["run_id"]:
        raise ValueError("--run-dir must identify a registered tune stage run")
    require_supported_setting(snapshot)
    config = snapshot["config"]
    experiment_id = run["experiment_id"]
    experiment_dir = Path(config["experiment"]["records_root"]).resolve() / experiment_id
    if run_dir != experiment_dir / "stage-runs" / "tune" / run["run_id"]:
        raise ValueError("Tune run directory disagrees with its frozen records_root")
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    if (experiment["experiment_id"] != experiment_id or experiment["status"] != "active"
            or (run["status"] not in ("planned", "running")
                and not (recovery_source and run["status"] in ("failed", "cancelled")))
            or snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or config["experiment"] != experiment_snapshot["config"]["experiment"]
            or not any(item == {"stage": "tune", "run_id": run["run_id"],
                               "path": run_dir.relative_to(experiment_dir).as_posix()}
                       for item in (experiment["stage_runs"] + experiment.get("retired_stage_runs", [])
                                    if recovery_source else experiment["stage_runs"]))):
        raise ValueError("Tune run is not an active child of its frozen experiment")
    if run["plan"] != "plan.json":
        raise ValueError("Tune orchestration requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    if (plan.get("stage") != "tune" or plan.get("setting") != config["experiment"]["setting"]
            or plan.get("config_fingerprint") != run["config_fingerprint"]
            or run["config_fingerprint"] != snapshot["config_fingerprint"]):
        raise ValueError("Tune plan does not match its frozen run configuration")

    setting = config["experiment"]["setting"]
    transfer_jobs = {}
    if setting == "multitask":
        expected = {"tune-whole": "whole"}
    elif setting == "singletask":
        expected = {"tune-{}".format(task): task for task in TASKS}
    elif setting == "cl":
        order = config["experiment"]["task_order"]
        if len(order) != len(TASKS) or set(order) != set(TASKS):
            raise ValueError("CL requires each of the six tasks exactly once")
        expected = {"tune-{:02d}-{}".format(index, task): task
                    for index, task in enumerate(order, 1)}
    elif setting in ("transfer_round1", "transfer_round2"):
        expected = {}
        for task in TASKS:
            job_id = "tune-round1-{}".format(task)
            expected[job_id] = task
            transfer_jobs[job_id] = (1, task, task)
        for source in TASKS:
            for target in TASKS:
                if source == target:
                    continue
                job_id = "tune-round2-{}-to-{}".format(source, target)
                expected[job_id] = target
                transfer_jobs[job_id] = (2, source, target)
    else:
        raise ValueError("Unsupported high-level tune setting: " + setting)
    planned = plan.get("jobs")
    if not isinstance(planned, list) or {item.get("job_id") for item in planned} != set(expected):
        raise ValueError("Tune plan has unexpected or missing low-level jobs")
    if len(planned) != len(expected):
        raise ValueError("Tune plan contains duplicate jobs")
    if {item["job_id"]: item.get("task") for item in planned} != expected:
        raise ValueError("Tune plan task assignment does not match its setting")
    if setting in ("cl", "transfer_round1", "transfer_round2") and [
            item["job_id"] for item in planned] != list(expected):
        raise ValueError("Tune plan order differs from its frozen setting")
    by_id = {item["job_id"]: item for item in planned}
    previous = None
    for job_id, task in expected.items():
        planned_job = by_id[job_id]
        depends_on, requires = [], []
        lineage = {"train_task": task, "initialization": "base_model",
                   "training_state": "fresh_optimizer_scheduler_epoch_global_step"}
        output_slot = task
        if setting == "cl":
            lineage["parent_job"] = previous
            if previous is not None:
                depends_on = [previous]
                requires = [{"stage": "tune", "job_id": previous, "artifact_kind": "adapter",
                             "slot": expected[previous]}]
                lineage["initialization"] = "previous_adapter_weights"
        elif setting in ("transfer_round1", "transfer_round2"):
            round_number, source, target = transfer_jobs[job_id]
            if round_number == 1:
                lineage = {"round": 1, "train_task": target, "initialization": "base_model",
                           "training_state": "fresh_optimizer_scheduler_epoch_global_step"}
                output_slot = "round1-{}".format(target)
            else:
                parent = "tune-round1-{}".format(source)
                depends_on = [parent]
                requires = [{"stage": "tune", "job_id": parent, "artifact_kind": "adapter",
                             "slot": "round1-{}".format(source)}]
                lineage = {"round": 2, "source_task": source, "train_task": target,
                           "initialization": "round1_adapter_weights", "parent_job": parent,
                           "training_state": "fresh_optimizer_scheduler_epoch_global_step"}
                output_slot = "round2-{}-to-{}".format(source, target)
        if (planned_job.get("depends_on", []) != depends_on
                or planned_job.get("requires", []) != requires
                or planned_job.get("lineage") != lineage
                or planned_job.get("output") != {"artifact_kind": "adapter", "slot": output_slot}):
            raise ValueError("{} has an invalid training dependency or lineage".format(job_id))
        previous = job_id
        annotation = [item for item in planned_job.get("input_specs", [])
                      if item.get("kind") == "annotation"]
        if (len(annotation) != 1 or annotation[0].get("metadata") !=
                {"task": expected[job_id], "split": "train"}):
            raise ValueError("{} must bind exactly its planned COSINE train split".format(job_id))
        annotation_role = "annotation-{}-train".format(expected[job_id])
        expected_specs = {
            "base-model": {"role": "base-model", "kind": "base_model",
                            "path": config["experiment"]["base_model"],
                            "source": "configuration:experiment.base_model", "metadata": {}},
            "processor": {"role": "processor", "kind": "processor",
                            "path": config["experiment"]["processor_path"],
                            "source": "configuration:experiment.processor_path", "metadata": {}},
            annotation_role: {"role": annotation_role, "kind": "annotation",
                                      "path": str(Path(config["experiment"]["dataset_root"])
                                                     / expected[job_id] / "train.json"),
                                      "source": "configuration:experiment.dataset_root",
                                      "metadata": {"task": expected[job_id], "split": "train"}},
        }
        specs = planned_job.get("input_specs", [])
        if (len(specs) != len(expected_specs)
                or {item.get("role"): item for item in specs} != expected_specs):
            raise ValueError("{} must bind exactly its configured base model, processor and train annotation".format(job_id))

    registered = {item["job_id"]: item for item in run["jobs"]}
    if set(registered) != set(expected) or len(run["jobs"]) != len(expected):
        raise ValueError("Registered low-level jobs do not match the frozen plan")
    jobs = {}
    for job_id, task in expected.items():
        directory = run_dir / "jobs" / job_id
        job, job_snapshot = load_job(directory)
        if (registered[job_id].get("path") != "jobs/" + job_id
                or job["job_id"] != job_id or job["task"] != task
                or job["stage"] != "tune" or job["run_id"] != run["run_id"]
                or job["experiment_id"] != experiment_id
                or job_snapshot["layers"].get("parent_snapshot") != snapshot["snapshot_fingerprint"]):
            raise ValueError("Low-level job identity or frozen configuration is inconsistent: " + job_id)
        resolved = resolve_job_config(snapshot, "tune", job_id,
                                      job_snapshot["layers"].get("job"),
                                      job_snapshot["layers"].get("job_cli"))
        if (resolved["config_fingerprint"] != job["config_fingerprint"]
                or job_snapshot["config"]["experiment"] != config["experiment"]):
            raise ValueError("Job configuration cannot be reproduced from the frozen parent: " + job_id)
        defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                    "input_specs": [], "lineage": {}, "output": None}
        if any(job["output_spec" if name == "output" else name] != by_id[job_id].get(name, default)
               for name, default in defaults.items()):
            raise ValueError("Job differs from the frozen tune plan: " + job_id)
        _training_policy(job)
        expected_output = (Path(config["experiment"]["checkpoints_root"])
                           / experiment_id / run["run_id"] / job_id)
        if Path(job["output_dir"]) != expected_output:
            raise ValueError("Job output directory disagrees with the frozen checkpoints_root: " + job_id)
        allowed_statuses = ("planned", "succeeded", "failed", "cancelled") if recovery_source else ("planned", "succeeded")
        if job["status"] not in allowed_statuses:
            raise ValueError("Job {} is {}; do not relaunch active or terminal attempts".format(
                job_id, job["status"]))
        info = {"directory": directory, "task": task, "snapshot": job_snapshot,
                "depends_on": list(job["depends_on"]), "initial_status": job["status"]}
        if job["status"] == "planned":
            if job["artifacts"]:
                raise ValueError("An unexecuted planned job cannot have outputs: " + job_id)
            # Dependent jobs are not ready yet. Repeat low-level checks just
            # before launch, after the parent executor has exited.
            if not recovery_source and not job["depends_on"]:
                _tune_job_context(directory)
            if ((directory / "execution.json").exists() or Path(job["output_dir"]).exists()
                    or Path(job["output_dir"]).is_symlink()):
                raise ValueError("Job {} already has execution evidence or an output; review it explicitly".format(job_id))
            job_config = job_snapshot["config"]
            job_run = job_config["tune"]["run"]
            python = Path(job_run["python"])
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError("Job Python is not executable: {}".format(python))
            restore_checkpoints = [item for item in job["inputs"].values()
                                   if item["kind"] == "checkpoint"]
            if restore_checkpoints:
                required_world_size = restore_checkpoints[0]["metadata"].get("world_size")
            else:
                required_world_size = None
            info["resources"] = []
            for index, group in enumerate(job_run["gpu_groups"]):
                if required_world_size is not None and len(group) != required_world_size:
                    continue
                port = job_run["master_port"] + index
                if port > 65535:
                    raise ValueError("Job {} has master_port + GPU group index above 65535".format(job_id))
                info["resources"].append({"group_index": index, "physical_gpu_ids": list(group),
                                          "port": (job_run["master_addr"], port)})
            if not info["resources"]:
                raise ValueError("No frozen GPU group matches the recovery source assignment: " + job_id)
        elif job["status"] == "succeeded":
            info["adapter"] = _successful_adapter(directory, job)
        jobs[job_id] = info
    for job_id, info in jobs.items():
        job, _ = load_job(info["directory"])
        if not info["depends_on"]:
            continue
        if info["initial_status"] == "succeeded":
            if any(jobs[parent]["initial_status"] != "succeeded" for parent in info["depends_on"]):
                raise ValueError("Succeeded jobs require every dependency to have succeeded: " + job_id)
            info["parent_adapter"] = _parent_adapter(job, jobs, require_bound=True)
        elif job["inputs"] and info["depends_on"]:
            # A previously launched dependent attempt must have been bound only
            # after its parent succeeded; preserve that lineage for recovery.
            parent = jobs[info["depends_on"][0]]
            if parent["initial_status"] == "succeeded":
                info["parent_adapter"] = _parent_adapter(job, jobs, require_bound=True)
    return run_dir, run, snapshot, setting, jobs


def _training_policy(job):
    """Distinguish cross-task weight inheritance from explicitly bound recovery."""
    adapters = [item for item in job["inputs"].values() if item["kind"] == "adapter"]
    checkpoints = [item for item in job["inputs"].values() if item["kind"] == "checkpoint"]
    chained = bool(job["depends_on"])
    resume = job["resume_training_state"]
    if not chained and adapters:
        raise ValueError("Independent tune jobs cannot inherit another task's adapter: " + job["job_id"])
    if chained and job["inputs"] and len(adapters) != 1:
        raise ValueError("A bound dependent job must contain exactly one parent adapter: " + job["job_id"])
    if type(resume) is not bool or len(checkpoints) > 1 or bool(checkpoints) != resume:
        raise ValueError("Training-state recovery requires one explicitly bound checkpoint: " + job["job_id"])
    policy = RESTORE if resume else CONTINUE if chained else FRESH
    if job["training_state_policy"] != policy:
        raise ValueError("Training-state policy disagrees with the frozen dependency: " + job["job_id"])


def _parent_adapter(job, jobs, *, require_bound=False):
    """Resolve only the audited final adapter from the frozen parent job."""
    parent_id, = job["depends_on"]
    parent_info = jobs[parent_id]
    if "adapter" not in parent_info:
        raise ValueError("Parent adapter has not passed its output audit: " + parent_id)
    parent_job, _ = load_job(parent_info["directory"])
    parent = _successful_adapter(parent_info["directory"], parent_job)
    if parent != parent_info["adapter"]:
        raise ValueError("Parent adapter registration changed after its output audit: " + parent_id)
    role = "parent-" + parent_info["task"]
    if job["inputs"] or require_bound:
        adapters = [(key, item) for key, item in job["inputs"].items() if item["kind"] == "adapter"]
        if len(adapters) != 1:
            raise ValueError("Dependent job lacks its exact bound parent adapter: " + job["job_id"])
        role, manifest = adapters[0]
        check_manifest(manifest)
        if manifest != read_json(parent["manifest"]):
            raise ValueError("Bound adapter differs from the registered parent output: " + job["job_id"])
    return {**parent, "input_role": role}


def _prepare_launch(info, jobs, resource):
    job, snapshot, experiment_dir, _ = _tune_job_context(info["directory"])
    if (snapshot["snapshot_fingerprint"] != info["snapshot"]["snapshot_fingerprint"]
            or job["depends_on"] != info["depends_on"]):
        raise ValueError("Frozen job changed while waiting for its dependencies")
    _training_policy(job)
    info["input_manifests"] = {}
    parent = None
    if info["depends_on"]:
        parent = _parent_adapter(job, jobs)
        info["parent_adapter"] = parent
        if not job["inputs"]:
            info["input_manifests"][parent["input_role"]] = parent["manifest"]

    source_dir = job.get("recovery_source_job_dir")
    if source_dir:
        source_dir = Path(source_dir).resolve(strict=True)
        source_job, _ = load_job(source_dir)
        inputs = dict(job.get("recovery_external_inputs", {}))
        if parent is not None:
            inputs[parent["input_role"]] = parent["manifest"]
        checkpoint = None
        if not job["inputs"] and source_job["status"] in ("failed", "cancelled"):
            candidate = _recovery_checkpoint(source_dir, source_job,
                                             {"job": job, "snapshot": snapshot,
                                              "directory": info["directory"], "task": job["task"]},
                                             inputs)
            if (candidate is not None
                    and candidate["metadata"].get("world_size") != len(resource["physical_gpu_ids"])):
                return "checkpoint_world_size_mismatch"
            checkpoint = candidate
        if checkpoint is not None:
            inputs["resume-checkpoint"] = checkpoint
        if not job["inputs"]:
            job = bind_inputs(info["directory"], inputs,
                              resume_training_state=checkpoint is not None)
        # The recovery job now owns frozen input manifests; do not pass a second
        # set of CLI inputs to the low-level executor.
        info["input_manifests"] = {}
        info["restore_checkpoint"] = checkpoint
        info["recovery_fallback"] = (None if checkpoint else
                                       ("continue_parent_adapter" if parent else "fresh_from_base"))
        if _can_reuse_success(source_dir, source_job, {**job, "snapshot": snapshot}, inputs):
            manifest = import_recovery_adapter(info["directory"], source_dir, inputs)
            info["adapter"] = {"manifest": str(info["directory"] / "artifacts/adapter.json"),
                                "path": manifest["path"], "fingerprint": manifest["fingerprint"],
                                "producer": manifest["producer"],
                                "slot": manifest["metadata"]["slot"]}
            info["reused_successful_adapter"] = True
            return True
        info["reused_successful_adapter"] = False
        return False
    if info["depends_on"] and not job["inputs"]:
        info["input_manifests"][parent["input_role"]] = parent["manifest"]
    return False


def _successful_adapter(job_dir, job, *, verify_contents=True):
    if job["status"] != "succeeded":
        raise ValueError("Job has not succeeded: " + job["job_id"])
    outputs = [item for item in job["artifacts"] if item["kind"] == "adapter"]
    if len(outputs) != 1:
        raise ValueError("A succeeded tune job must have exactly one registered adapter")
    artifact = outputs[0]
    if artifact["path"] != "artifacts/" + artifact["role"] + ".json":
        raise ValueError("Adapter manifest path does not match its registered role")
    manifest_path = job_dir / artifact["path"]
    manifest = check_manifest(read_json(manifest_path), verify_contents=verify_contents)
    if (manifest["kind"] != "adapter" or manifest["fingerprint"] != artifact["fingerprint"]
            or manifest["requested_path"] != str(Path(job["output_dir"]) / "adapter")
            or manifest["path"] != str((Path(job["output_dir"]) / "adapter").resolve(strict=True))
            or manifest["producer"] != {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
            or manifest["metadata"].get("slot") != job["output_spec"]["slot"]
            or manifest["metadata"].get("training_task") != job["task"]
            or manifest["metadata"].get("config_fingerprint") != job["config_fingerprint"]):
        raise ValueError("Adapter does not describe the exact successful tune job")
    if "parent_job" in job["lineage"] or job["lineage"].get("round") in (1, 2):
        _training_policy(job)
        if not job["inputs"]:
            raise ValueError("Succeeded tune job lacks its bound inputs")
        metadata = manifest["metadata"]
        inherited = {role: item["fingerprint"] for role, item in job["inputs"].items()
                     if item["kind"] in ("adapter", "checkpoint")}
        if metadata.get("lineage") != job["lineage"] or metadata.get("inherited_inputs") != inherited:
            raise ValueError("Adapter metadata disagrees with its recorded weight lineage")
        reused = job.get("reused_from")
        if reused is None:
            if metadata.get("training_state_policy") != job["training_state_policy"]:
                raise ValueError("Adapter training-state policy differs from its producing job")
        else:
            producer = reused.get("job")
            if (not isinstance(producer, dict)
                    or set(producer) != {"experiment_id", "stage", "run_id", "job_id"}
                    or producer["stage"] != "tune"):
                raise ValueError("Reused adapter source identity is malformed")
            _, job_snapshot = load_job(job_dir)
            records_root = Path(job_snapshot["config"]["experiment"]["records_root"])
            source_dir = records_root / producer["experiment_id"] / "stage-runs" / "tune" / producer["run_id"] / "jobs" / producer["job_id"]
            source_job, _ = load_job(source_dir)
            source_artifact = next((item for item in source_job["artifacts"]
                                    if item["kind"] == "adapter"
                                    and item["fingerprint"] == reused.get("fingerprint")), None)
            if (source_job["status"] != "succeeded" or source_artifact is None
                    or any(source_job[key] != producer[key] for key in producer)
                    or source_job["task"] != job["task"]
                    or source_job["config_fingerprint"] != job["config_fingerprint"]
                    or source_job["lineage"] != job["lineage"]):
                raise ValueError("Reused adapter no longer matches its successful source job")
            source_manifest = check_manifest(read_json(source_dir / source_artifact["path"]),
                                             verify_contents=verify_contents)
            if (source_manifest["fingerprint"] != reused["fingerprint"]
                    or source_manifest["identity"] != manifest["identity"]
                    or source_manifest["metadata"].get("lineage") != job["lineage"]):
                raise ValueError("Reused adapter payload or lineage differs from its source")
    return {"manifest": str(manifest_path), "path": manifest["path"],
            "fingerprint": manifest["fingerprint"], "producer": manifest["producer"],
            "slot": manifest["metadata"]["slot"]}


@contextmanager
def _run_lock(run_dir):
    path = run_dir / ".tune-orchestrator.lock"
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another high-level tune run is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _mark(job_dir, status, message):
    current = record_status(job_dir)["status"]
    if current in ("planned", "running"):
        update_status(job_dir, status, message)


def _terminate(active):
    """Wait for low-level worker shutdown and checkpoint preservation before exiting."""
    for item in active.values():
        if item["process"].poll() is None:
            try:
                item["process"].send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
    # The low-level executor owns its torchrun session and cleanup. Do not kill
    # it on a timer while it is still terminating workers or hashing checkpoints.
    for item in active.values():
        try:
            item["process"].wait()
        finally:
            item["log"].close()


@contextmanager
def _signals():
    state = {"number": None}

    def interrupt(signum, frame):
        # Defer the exception to a scheduler boundary so Popen and registering
        # its handle cannot be interrupted between spawning and taking ownership.
        state["number"] = signum
    previous = {number: signal.signal(number, interrupt)
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _summary(run_dir, invocation_dir, run, setting, jobs, started_at, status, details):
    entries = []
    for job_id, info in jobs.items():
        record = record_status(info["directory"])
        entries.append({"job_id": job_id, "task": info["task"], "status": record["status"],
                        "config_fingerprint": record["config_fingerprint"],
                        "output_dir": record["output_dir"], "initial_status": info["initial_status"],
                        "returncode": info.get("returncode"), "log": info.get("log_path"),
                        "assignment": info.get("assignment"), "adapter": info.get("adapter"),
                        "depends_on": record["depends_on"], "lineage": record["lineage"],
                        "training_state_policy": record["training_state_policy"],
                        "resume_training_state": record["resume_training_state"],
                        "parent_adapter": info.get("parent_adapter"), "blocked_by": info.get("blocked_by", []),
                        "reused_from": record.get("reused_from"),
                        "reused_successful_adapter": info.get("reused_successful_adapter", False),
                        "restore_checkpoint": (info["restore_checkpoint"].get("fingerprint")
                                               if info.get("restore_checkpoint") else None),
                        "recovery_fallback": info.get("recovery_fallback"),
                        "artifacts": [{**artifact, "manifest": str(info["directory"] / artifact["path"])}
                                      for artifact in record["artifacts"]]})
    result = {
        "schema_version": 1, "kind": "tune_run_summary", "experiment_id": run["experiment_id"],
        "run_id": run["run_id"], "recovery_of": run.get("recovery_of"),
        "setting": setting, "status": status,
        "job_order": list(jobs), "task_order": [info["task"] for info in jobs.values()],
        "config_fingerprint": run["config_fingerprint"], "invocation_dir": str(invocation_dir),
        "started_at": started_at, "finished_at": utc_now(), "jobs": entries,
    }
    if details:
        result["details"] = details
    write_json_atomic(invocation_dir / "summary.json", result)
    write_json_atomic(run_dir / "tune-summary.json", result)
    return result


def _collect(active, jobs, failures, *, stop_message=None):
    """Collect exited executors before assigning their resources to another job."""
    completed = 0
    unsafe = []
    for job_id, item in list(active.items()):
        returncode = item["process"].poll()
        if returncode is None:
            continue
        item["log"].close()
        info = jobs[job_id]
        info["returncode"] = returncode
        child = record_status(info["directory"])
        output_error = None
        if returncode == 0 and child["status"] == "succeeded":
            try:
                adapter = _successful_adapter(info["directory"], child)
                if info["depends_on"]:
                    info["parent_adapter"] = _parent_adapter(child, jobs, require_bound=True)
                info["adapter"] = adapter
            except (OSError, ValueError, KeyError) as error:
                output_error = str(error)
        execution_path = info["directory"] / "execution.json"
        cleanup_error = (read_json(execution_path).get("cleanup_error")
                         if execution_path.is_file() else None)
        incomplete_shutdown = child["status"] == "running" or bool(cleanup_error)
        if returncode != 0 or child["status"] != "succeeded" or output_error or cleanup_error:
            message = "Low-level tune job exited {} with status {}".format(returncode, child["status"])
            if output_error:
                message += "; adapter audit failed: " + output_error
            if cleanup_error:
                message += "; worker cleanup failed: " + str(cleanup_error)
            if stop_message:
                message += "; " + stop_message
            _mark(info["directory"], "cancelled" if stop_message else "failed", message)
            failures.append({"job_id": job_id, "returncode": returncode, "message": message})
        status = record_status(info["directory"])["status"]
        item["launch"].update(returncode=returncode, finished_at=utc_now(), status=status)
        del active[job_id]
        completed += 1
        if incomplete_shutdown:
            # The launcher's own torchrun session may outlive an abrupt parent
            # exit. Stop this scheduler regardless of fail_fast.
            unsafe.append(job_id)
        print("Finished {}: {} (exit {})".format(job_id, status, returncode), flush=True)
    return completed, unsafe


def _cancel(pending, active, jobs, failures, message):
    _terminate(active)
    _collect(active, jobs, failures, stop_message=message)
    for job_id in list(pending):
        _mark(jobs[job_id]["directory"], "cancelled", message + "; job was not launched")
        pending.remove(job_id)


def _launch(job_id, info, resource, repo_root, low_level, invocation_dir, receipt, active):
    log_path = invocation_dir / (job_id + ".log")
    job_python = info["snapshot"]["config"]["tune"]["run"]["python"]
    command = [job_python, str(low_level), "--job-dir", str(info["directory"]),
               "--gpu-group-index", str(resource["group_index"])]
    for role, manifest in info.get("input_manifests", {}).items():
        command.extend(["--input", "{}={}".format(role, manifest)])
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, resource["physical_gpu_ids"])),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    launch = {"job_id": job_id, "command": command, "assignment": resource,
              "log": str(log_path), "started_at": utc_now(), "pid": None, "status": "launching",
              "depends_on": info["depends_on"], "parent_adapter": info.get("parent_adapter")}
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


def _external_inputs(job, cache):
    inputs = {}
    for spec in job["input_specs"]:
        cached = job["inputs"].get(spec["role"])
        if cached is not None:
            if (cached["kind"] != spec["kind"]
                    or cached["requested_path"] != str(Path(spec["path"]).expanduser().absolute())
                    or cached["external_source"] != spec["source"]
                    or cached["metadata"] != spec["metadata"]):
                raise ValueError("Frozen recovery input binding changed: " + spec["role"])
            verified_key = ("verified", cached["fingerprint"])
            if verified_key not in cache:
                check_manifest(cached, verify_contents=True)
                cache[verified_key] = cached
            inputs[spec["role"]] = cached
            continue
        key = (spec["kind"], spec["path"], spec["source"], tuple(sorted(spec["metadata"].items())))
        if key not in cache:
            cache[key] = make_manifest(spec["kind"], spec["path"], source=spec["source"],
                                       metadata=spec["metadata"])
        inputs[spec["role"]] = cache[key]
    return inputs


def _recovery_checkpoint(source_dir, source_job, target_job, inputs):
    """Return the newest registered checkpoint that matches this exact retry."""
    if source_job["status"] not in ("failed", "cancelled"):
        return None
    config = target_job["snapshot"]["config"]
    params = config["tune"]["params"]
    groups = config["tune"]["run"]["gpu_groups"]
    world_sizes = {len(group) for group in groups}
    source_receipt = read_json(source_dir / "execution.json") if (source_dir / "execution.json").is_file() else {}
    assigned = source_receipt.get("physical_gpu_ids")
    if isinstance(assigned, list) and assigned:
        source_world_size = len(assigned)
        if source_world_size not in world_sizes:
            return None
    deepspeed_sha = sha256_file(params["deepspeed"]) if params["deepspeed"] else None
    expected_inputs = {role: manifest["fingerprint"] for role, manifest in inputs.items()
                       if manifest["kind"] not in ("adapter", "checkpoint")}
    expected_parent = [item for item in inputs.values() if item["kind"] == "adapter"]
    candidates = []
    for item in source_job["artifacts"]:
        if item["kind"] != "checkpoint":
            continue
        try:
            manifest_path = source_dir / item["path"]
            manifest = check_manifest(read_json(manifest_path), verify_contents=True)
            if (manifest["kind"] != "checkpoint"
                    or manifest["producer"] != {key: source_job[key] for key in
                                                   ("experiment_id", "stage", "run_id", "job_id")}
                    or manifest["fingerprint"] != item["fingerprint"]):
                continue
            metadata = manifest["metadata"]
            if (metadata.get("training_task") != target_job["task"]
                    or metadata.get("config_fingerprint") != target_job["job"]["config_fingerprint"]
                    or metadata.get("lineage") != target_job["job"]["lineage"]
                    or metadata.get("input_fingerprints") != expected_inputs
                    or metadata.get("world_size") not in world_sizes
                    or metadata.get("deepspeed_sha256") != deepspeed_sha):
                continue
            source_parents = [value for value in source_job["inputs"].values()
                              if value["kind"] == "adapter"]
            if ([value["identity"] for value in source_parents]
                    != [value["identity"] for value in expected_parent]):
                continue
            layout = _checkpoint_state(manifest["path"], metadata["world_size"],
                                       params["deepspeed"], params["fp16"])
            recorded = metadata.get("checkpoint_state")
            legacy = {key: layout[key] for key in
                      ("global_step", "world_size", "backend", "state_files")}
            if recorded not in (layout, legacy) or metadata.get("global_step") != layout["global_step"]:
                continue
            candidates.append((layout["global_step"], manifest))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _can_reuse_success(source_dir, source_job, recovery_job, recovery_inputs):
    """Reuse only if frozen config, external inputs and parent weights still match."""
    if (source_job["status"] != "succeeded"
            or source_job["config_fingerprint"] != recovery_job["config_fingerprint"]
            or source_job["lineage"] != recovery_job["lineage"]):
        return False
    try:
        source_adapter = _successful_adapter(source_dir, source_job, verify_contents=True)
        metadata = check_manifest(read_json(source_adapter["manifest"]),
                                  verify_contents=True)["metadata"]
    except (OSError, ValueError, KeyError):
        return False
    current_external = {role: manifest["fingerprint"] for role, manifest in recovery_inputs.items()
                        if manifest["kind"] not in ("adapter", "checkpoint")}
    if metadata.get("input_fingerprints") != current_external:
        return False
    params = recovery_job["snapshot"]["config"]["tune"]["params"]
    deepspeed_sha = sha256_file(params["deepspeed"]) if params["deepspeed"] else None
    if metadata.get("deepspeed_sha256") != deepspeed_sha:
        return False
    source_parents = [item for item in source_job["inputs"].values() if item["kind"] == "adapter"]
    current_parents = [item for item in recovery_inputs.values() if item["kind"] == "adapter"]
    return [item["identity"] for item in source_parents] == [item["identity"] for item in current_parents]


def prepare_tune_recovery(source_run_dir):
    """Create and register a new recovery run without launching model work."""
    source_run_dir = Path(source_run_dir).expanduser().resolve(strict=True)
    source_context = _context(source_run_dir, recovery_source=True)
    _, source_run, snapshot, _, source_jobs = source_context
    experiment_dir = source_run_dir.parents[2]
    recovery_dir = create_recovery_stage_run(experiment_dir, source_run_dir)
    cache = {}
    activated = False
    try:
        for job_id, source_info in source_jobs.items():
            source_job, source_snapshot = load_job(source_info["directory"])
            fields = {"task": source_job["task"], "tasks": source_job["tasks"],
                      "depends_on": source_job["depends_on"], "requires": source_job["requires"],
                      "input_specs": source_job["input_specs"], "lineage": source_job["lineage"],
                      "output": source_job["output_spec"]}
            external_inputs = _external_inputs(source_job, cache)
            target_dir = create_job(
                recovery_dir, job_id,
                job_override=source_snapshot["layers"].get("job"),
                cli_override=source_snapshot["layers"].get("job_cli"),
                inputs={}, resume_training_state=False, **fields)
            set_recovery_source(target_dir, source_info["directory"], external_inputs)
        activate_recovery_stage_run(experiment_dir, source_run_dir, recovery_dir)
        activated = True
    except BaseException:
        import shutil
        for child in recovery_dir.glob("jobs/*/job.json"):
            try:
                job_record = read_json(child)
                output = Path(job_record["output_dir"])
                expected = (Path(snapshot["config"]["experiment"]["checkpoints_root"])
                            / source_run["experiment_id"] / recovery_dir.name / job_record["job_id"])
                if output == expected and output.is_symlink():
                    output.unlink()
                elif output == expected and output.is_dir():
                    shutil.rmtree(output)
            except (OSError, ValueError, KeyError):
                pass
        if not activated:
            shutil.rmtree(recovery_dir, ignore_errors=True)
        raise
    return recovery_dir


def execute_tune_run(run_dir):
    """Schedule frozen tune dependencies with resource and failure handling."""
    return run_logged("tune", run_dir, _execute_tune_run)


def _execute_tune_run(run_dir, invocation_dir):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    with _run_lock(run_dir), _signals() as signals:
        run_dir, run, snapshot, setting, jobs = _context(run_dir)
        config = snapshot["config"]
        tune_run = config["tune"]["run"]
        capacity = tune_run["max_parallel_jobs"]
        fail_fast = tune_run["fail_fast"]
        repo_root = Path(config["experiment"]["repo_root"])
        low_level = repo_root / "scripts" / "experiments" / "run_tune_job.py"
        if not low_level.is_file():
            raise ValueError("Low-level tune entry point is missing: {}".format(low_level))
        started_at = utc_now()
        receipt = {"schema_version": 1, "run_id": run["run_id"], "experiment_id": run["experiment_id"],
                   "recovery_of": run.get("recovery_of"),
                   "config_fingerprint": run["config_fingerprint"], "started_at": started_at,
                   "pid": os.getpid(), "status": "running", "max_parallel_jobs": capacity,
                   "fail_fast": fail_fast, "setting": setting, "job_order": list(jobs), "launches": []}
        write_json_atomic(invocation_dir / "execution.json", receipt)
        pending = [job_id for job_id in jobs if jobs[job_id]["initial_status"] == "planned"]
        active, failures = {}, []
        interrupted = False
        try:
            if signals["number"] is not None:
                raise InterruptedError("Tune orchestrator interrupted during preparation")
            if run["status"] == "planned":
                update_status(run_dir, "running", "High-level {} tune run started".format(setting))
            while pending or active:
                if signals["number"] is not None:
                    raise InterruptedError("Tune orchestrator received signal {}".format(signals["number"]))
                completed, unsafe = _collect(active, jobs, failures)
                if completed:
                    write_json_atomic(invocation_dir / "execution.json", receipt)
                if unsafe:
                    _cancel(pending, active, jobs, failures,
                            "Low-level worker shutdown is unconfirmed for {}; review cleanup".format(unsafe))
                    break
                if fail_fast and failures:
                    _cancel(pending, active, jobs, failures, "Stopped after another tune job failed")
                    break
                if pending:
                    busy_gpus = {gpu for item in active.values() for gpu in item["resource"]["physical_gpu_ids"]}
                    busy_ports = {item["resource"]["port"] for item in active.values()}
                    for job_id in list(pending):
                        if signals["number"] is not None:
                            raise InterruptedError("Tune orchestrator interrupted before launch")
                        info = jobs[job_id]
                        if len(active) >= capacity:
                            break
                        failed_ids = {item.get("job_id") for item in failures}
                        blocked = [parent for parent in info["depends_on"]
                                   if parent in failed_ids or record_status(jobs[parent]["directory"])["status"]
                                   in ("failed", "cancelled")]
                        if blocked:
                            message = "Required parent tune job failed or was cancelled: {}".format(blocked)
                            info["blocked_by"] = blocked
                            _mark(info["directory"], "cancelled", message + "; job was not launched")
                            failures.append({"job_id": job_id, "blocked_by": blocked, "message": message})
                            pending.remove(job_id)
                            if fail_fast:
                                break
                            continue
                        if any(parent in pending or parent in active or "adapter" not in jobs[parent]
                               for parent in info["depends_on"]):
                            continue
                        resources = [item for item in info["resources"]
                                     if not busy_gpus.intersection(item["physical_gpu_ids"])
                                     and item["port"] not in busy_ports]
                        if not resources:
                            continue
                        deferred = False
                        launched = False
                        for resource in resources:
                            try:
                                reused = _prepare_launch(info, jobs, resource)
                                if reused == "checkpoint_world_size_mismatch":
                                    deferred = True
                                    continue
                                if reused:
                                    pending.remove(job_id)
                                    record = load_job(info["directory"])[0]
                                    receipt["launches"].append({"job_id": job_id, "status": "reused",
                                                                "reused_from": record.get("reused_from"),
                                                                "assignment": resource})
                                    write_json_atomic(invocation_dir / "execution.json", receipt)
                                    print("Reused verified adapter for {}".format(job_id), flush=True)
                                    launched = True
                                    break
                                _launch(job_id, info, resource, repo_root, low_level, invocation_dir, receipt, active)
                            except (OSError, ValueError, KeyError) as error:
                                pending.remove(job_id)
                                _mark(info["directory"], "failed", "Cannot launch low-level executor: {}".format(error))
                                failures.append({"job_id": job_id, "message": str(error),
                                                 "traceback": traceback.format_exc()})
                                write_json_atomic(invocation_dir / "execution.json", receipt)
                                launched = True
                                break
                            pending.remove(job_id)
                            write_json_atomic(invocation_dir / "execution.json", receipt)
                            busy_gpus.update(resource["physical_gpu_ids"])
                            busy_ports.add(resource["port"])
                            launched = True
                            break
                        if not launched and deferred:
                            continue
                        if fail_fast and failures:
                            break
                        if launched and len(active) >= capacity:
                            break

                if active:
                    time.sleep(0.1)
                elif pending and not (fail_fast and failures):
                    raise ValueError("Pending tune jobs have no runnable dependencies or GPU assignment: {}".format(pending))
            if signals["number"] is not None:
                raise InterruptedError("Tune orchestrator interrupted during execution")
        except (KeyboardInterrupt, InterruptedError):
            interrupted = True
            _cancel(pending, active, jobs, failures, "High-level tune run interrupted")
        except BaseException as error:
            _cancel(pending, active, jobs, failures, "Scheduler stopped: {}".format(error))
            failures.append({"message": "{}: {}".format(type(error).__name__, error),
                             "traceback": traceback.format_exc()})

        child_statuses = [record_status(info["directory"])["status"] for info in jobs.values()]
        final_status = ("cancelled" if interrupted else
                        "succeeded" if not failures and child_statuses and all(item == "succeeded" for item in child_statuses)
                        else "failed")
        summary = _summary(run_dir, invocation_dir, run, setting, jobs, started_at, final_status, failures)
        update_status(run_dir, final_status,
                      "All {} tune jobs completed".format(len(jobs)) if final_status == "succeeded"
                      else "High-level tune run {}".format(final_status))
        receipt.update(status=final_status, finished_at=utc_now(), summary=str(invocation_dir / "summary.json"))
        write_json_atomic(invocation_dir / "execution.json", receipt)
        experiment_dir = run_dir.parents[2]
        experiment, _ = load_experiment(experiment_dir)
        if (final_status == "succeeded" and experiment["status"] == "active" and experiment["stage_runs"]
                and all(record_status(experiment_dir / item["path"])["status"] == "succeeded"
                        for item in experiment["stage_runs"])):
            update_status(experiment_dir, "succeeded", "All registered stage runs completed")
        return summary
