"""Run one frozen Multitask, Singletask or CL tune stage over its low-level jobs."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import check_manifest
from .common import TASKS, new_id, read_json, utc_now, write_json_atomic
from .config import resolve_job_config
from .records import (
    load_experiment, load_job, load_stage_run, record_status, update_status,
)
from .tune_executor import CONTINUE, FRESH, RESTORE, _context as _tune_job_context

SUPPORTED_SETTINGS = ("multitask", "singletask", "cl")


def require_supported_setting(snapshot):
    experiment = snapshot["config"]["experiment"]
    if experiment["setting"] not in SUPPORTED_SETTINGS:
        raise ValueError("High-level tune currently supports only multitask, singletask and cl")
    if "tune" not in experiment["enabled_stages"]:
        raise ValueError("The experiment must enable the tune stage")


def _context(run_dir):
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
            or run["status"] not in ("planned", "running")
            or snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or config["experiment"] != experiment_snapshot["config"]["experiment"]
            or {"stage": "tune", "run_id": run["run_id"],
                "path": run_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]):
        raise ValueError("Tune run is not an active child of its frozen experiment")
    if run["plan"] != "plan.json":
        raise ValueError("Tune orchestration requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    if (plan.get("stage") != "tune" or plan.get("setting") != config["experiment"]["setting"]
            or plan.get("config_fingerprint") != run["config_fingerprint"]
            or run["config_fingerprint"] != snapshot["config_fingerprint"]):
        raise ValueError("Tune plan does not match its frozen run configuration")

    setting = config["experiment"]["setting"]
    if setting == "multitask":
        expected = {"tune-whole": "whole"}
    elif setting == "singletask":
        expected = {"tune-{}".format(task): task for task in TASKS}
    else:
        order = config["experiment"]["task_order"]
        if len(order) != len(TASKS) or set(order) != set(TASKS):
            raise ValueError("CL requires each of the six tasks exactly once")
        expected = {"tune-{:02d}-{}".format(index, task): task
                    for index, task in enumerate(order, 1)}
    planned = plan.get("jobs")
    if not isinstance(planned, list) or {item.get("job_id") for item in planned} != set(expected):
        raise ValueError("Tune plan has unexpected or missing low-level jobs")
    if len(planned) != len(expected):
        raise ValueError("Tune plan contains duplicate jobs")
    if {item["job_id"]: item.get("task") for item in planned} != expected:
        raise ValueError("Tune plan task assignment does not match its setting")
    if setting == "cl" and [item["job_id"] for item in planned] != list(expected):
        raise ValueError("CL plan order differs from the frozen task_order")
    by_id = {item["job_id"]: item for item in planned}
    previous = None
    for job_id, task in expected.items():
        planned_job = by_id[job_id]
        depends_on, requires = [], []
        lineage = {"train_task": task, "initialization": "base_model",
                   "training_state": "fresh_optimizer_scheduler_epoch_global_step"}
        if setting == "cl":
            lineage["parent_job"] = previous
            if previous is not None:
                depends_on = [previous]
                requires = [{"stage": "tune", "job_id": previous, "artifact_kind": "adapter",
                             "slot": expected[previous]}]
                lineage["initialization"] = "previous_adapter_weights"
        if (planned_job.get("depends_on", []) != depends_on
                or planned_job.get("requires", []) != requires
                or planned_job.get("lineage") != lineage
                or planned_job.get("output") != {"artifact_kind": "adapter", "slot": task}):
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
        if job["status"] not in ("planned", "succeeded"):
            raise ValueError("Job {} is {}; do not relaunch active or terminal attempts".format(
                job_id, job["status"]))
        info = {"directory": directory, "task": task, "snapshot": job_snapshot,
                "depends_on": list(job["depends_on"]), "initial_status": job["status"]}
        if job["status"] == "planned":
            if job["artifacts"]:
                raise ValueError("An unexecuted planned job cannot have outputs: " + job_id)
            # Later CL stages are not ready yet. Repeat the low-level checks
            # immediately before launch, after the parent executor has exited.
            if not job["depends_on"]:
                _tune_job_context(directory)
            if ((directory / "execution.json").exists() or Path(job["output_dir"]).exists()
                    or Path(job["output_dir"]).is_symlink()):
                raise ValueError("Job {} already has execution evidence or an output; review it explicitly".format(job_id))
            job_config = job_snapshot["config"]
            job_run = job_config["tune"]["run"]
            python = Path(job_run["python"])
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError("Job Python is not executable: {}".format(python))
            info["resources"] = []
            for index, group in enumerate(job_run["gpu_groups"]):
                port = job_run["master_port"] + index
                if port > 65535:
                    raise ValueError("Job {} has master_port + GPU group index above 65535".format(job_id))
                info["resources"].append({"group_index": index, "physical_gpu_ids": list(group),
                                          "port": (job_run["master_addr"], port)})
        else:
            info["adapter"] = _successful_adapter(directory, job)
        jobs[job_id] = info
    for job_id, info in jobs.items():
        job, _ = load_job(info["directory"])
        if not info["depends_on"]:
            continue
        if info["initial_status"] == "succeeded":
            if any(jobs[parent]["initial_status"] != "succeeded" for parent in info["depends_on"]):
                raise ValueError("Succeeded CL stages must form a continuous prefix: " + job_id)
            info["parent_adapter"] = _parent_adapter(job, jobs, require_bound=True)
        elif job["inputs"]:
            # A manually bound job must still use this run's exact parent.
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
        raise ValueError("A bound CL stage must contain exactly one parent adapter: " + job["job_id"])
    if type(resume) is not bool or len(checkpoints) > 1 or bool(checkpoints) != resume:
        raise ValueError("Training-state recovery requires one explicitly bound checkpoint: " + job["job_id"])
    policy = RESTORE if resume else CONTINUE if chained else FRESH
    if job["training_state_policy"] != policy:
        raise ValueError("Training-state policy disagrees with the frozen dependency: " + job["job_id"])


def _parent_adapter(job, jobs, *, require_bound=False):
    """Resolve only the audited final adapter from the preceding frozen CL job."""
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
            raise ValueError("CL stage lacks its exact bound parent adapter: " + job["job_id"])
        role, manifest = adapters[0]
        check_manifest(manifest)
        if manifest != read_json(parent["manifest"]):
            raise ValueError("Bound adapter differs from the preceding stage's registered output: " + job["job_id"])
    return {**parent, "input_role": role}


def _prepare_launch(info, jobs):
    job, snapshot, _, _ = _tune_job_context(info["directory"])
    if (snapshot["snapshot_fingerprint"] != info["snapshot"]["snapshot_fingerprint"]
            or job["depends_on"] != info["depends_on"]):
        raise ValueError("Frozen job changed while waiting for its dependencies")
    _training_policy(job)
    info["input_manifests"] = {}
    if info["depends_on"]:
        parent = _parent_adapter(job, jobs)
        info["parent_adapter"] = parent
        if not job["inputs"]:
            info["input_manifests"][parent["input_role"]] = parent["manifest"]


def _successful_adapter(job_dir, job):
    if job["status"] != "succeeded":
        raise ValueError("Job has not succeeded: " + job["job_id"])
    outputs = [item for item in job["artifacts"] if item["kind"] == "adapter"]
    if len(outputs) != 1:
        raise ValueError("A succeeded tune job must have exactly one registered adapter")
    artifact = outputs[0]
    if artifact["path"] != "artifacts/" + artifact["role"] + ".json":
        raise ValueError("Adapter manifest path does not match its registered role")
    manifest_path = job_dir / artifact["path"]
    manifest = check_manifest(read_json(manifest_path), verify_contents=True)
    if (manifest["kind"] != "adapter" or manifest["fingerprint"] != artifact["fingerprint"]
            or manifest["requested_path"] != str(Path(job["output_dir"]) / "adapter")
            or manifest["path"] != str((Path(job["output_dir"]) / "adapter").resolve(strict=True))
            or manifest["producer"] != {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
            or manifest["metadata"].get("slot") != job["output_spec"]["slot"]
            or manifest["metadata"].get("training_task") != job["task"]
            or manifest["metadata"].get("config_fingerprint") != job["config_fingerprint"]):
        raise ValueError("Adapter does not describe the exact successful tune job")
    if "parent_job" in job["lineage"]:
        _training_policy(job)
        if not job["inputs"]:
            raise ValueError("Succeeded CL stage lacks its bound inputs")
        metadata = manifest["metadata"]
        inherited = {role: item["fingerprint"] for role, item in job["inputs"].items()
                     if item["kind"] in ("adapter", "checkpoint")}
        if (metadata.get("lineage") != job["lineage"]
                or metadata.get("training_state_policy") != job["training_state_policy"]
                or metadata.get("inherited_inputs") != inherited):
            raise ValueError("CL adapter metadata disagrees with its recorded weight lineage")
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
                        "artifacts": [{**artifact, "manifest": str(info["directory"] / artifact["path"])}
                                      for artifact in record["artifacts"]]})
    result = {
        "schema_version": 1, "kind": "tune_run_summary", "experiment_id": run["experiment_id"],
        "run_id": run["run_id"], "setting": setting, "status": status,
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


def execute_tune_run(run_dir):
    """Schedule independent jobs or a serial CL chain with frozen resources."""
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
        invocation_dir = run_dir / "orchestrations" / new_id("tune")
        invocation_dir.mkdir(parents=True, exist_ok=False)
        receipt = {"schema_version": 1, "run_id": run["run_id"], "experiment_id": run["experiment_id"],
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
                            message = "Required preceding tune stage failed or was cancelled: {}".format(blocked)
                            info["blocked_by"] = blocked
                            _mark(info["directory"], "cancelled", message + "; job was not launched")
                            failures.append({"job_id": job_id, "blocked_by": blocked, "message": message})
                            pending.remove(job_id)
                            continue
                        if any(parent in pending or parent in active or "adapter" not in jobs[parent]
                               for parent in info["depends_on"]):
                            continue
                        resource = next((item for item in info["resources"]
                                         if not busy_gpus.intersection(item["physical_gpu_ids"])
                                         and item["port"] not in busy_ports), None)
                        if resource is None:
                            continue
                        try:
                            _prepare_launch(info, jobs)
                            _launch(job_id, info, resource, repo_root, low_level, invocation_dir, receipt, active)
                        except (OSError, ValueError, KeyError) as error:
                            pending.remove(job_id)
                            _mark(info["directory"], "failed", "Cannot launch low-level executor: {}".format(error))
                            failures.append({"job_id": job_id, "message": str(error)})
                            write_json_atomic(invocation_dir / "execution.json", receipt)
                            if fail_fast:
                                break
                            continue
                        pending.remove(job_id)
                        write_json_atomic(invocation_dir / "execution.json", receipt)
                        busy_gpus.update(resource["physical_gpu_ids"])
                        busy_ports.add(resource["port"])
                        if len(active) >= capacity:
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
            failures.append({"message": "{}: {}".format(type(error).__name__, error)})

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
