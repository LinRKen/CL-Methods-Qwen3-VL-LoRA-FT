"""Filesystem-backed experiment, stage-run and job records."""

from copy import deepcopy
from pathlib import Path

from .artifacts import ARTIFACT_KINDS, check_manifest
from .common import TASKS, fingerprint, new_id, read_json, record_lock, require_id, utc_now, write_json_atomic
from .config import resolve_job_config, resolve_run_config, validate_snapshot

RECORD_SCHEMA_VERSION = 1
TERMINAL = {"succeeded", "failed", "cancelled"}
TRANSITIONS = {
    "active": TERMINAL,
    "planned": {"running", "failed", "cancelled"},
    "running": TERMINAL,
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}
STAGE_OUTPUT_ROOTS = {"tune": "checkpoints_root", "infer": "infer_root", "eval": "eval_root"}
STAGE_ARTIFACTS = {
    "tune": {"adapter", "checkpoint", "summary"},
    "infer": {"prediction", "summary"},
    "eval": {"judge_raw", "judge_scored", "metrics", "summary"},
}
PRIMARY_ARTIFACTS = {"tune": "adapter", "infer": "prediction", "eval": "metrics"}


def _read_record(directory, filename):
    path = Path(directory) / filename
    value = read_json(path)
    if not isinstance(value, dict) or value.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed record: {}".format(path))
    return path, value


def _load_snapshot(directory, record):
    name = Path(record["config_snapshot"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Snapshot must remain inside its record directory")
    snapshot = read_json(directory / name)
    validate_snapshot(snapshot)
    if snapshot["config_fingerprint"] != record["config_fingerprint"]:
        raise ValueError("Record and configuration snapshot disagree")
    return snapshot


def _job_fields(task, tasks, depends_on, requires, input_specs, lineage, output):
    if task is not None and (not isinstance(task, str) or task not in (*TASKS, "whole")):
        raise ValueError("Unknown task: {}".format(task))
    if tasks is not None and (
        not isinstance(tasks, list) or not tasks
        or any(not isinstance(item, str) or item not in TASKS for item in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("tasks must list distinct COSINE test tasks")
    depends_on = [] if depends_on is None else deepcopy(depends_on)
    if not isinstance(depends_on, list) or any(not isinstance(item, str) for item in depends_on):
        raise ValueError("depends_on must list local job IDs")
    for identifier in depends_on:
        require_id(identifier)
    if len(set(depends_on)) != len(depends_on):
        raise ValueError("Duplicate job dependency")
    requires = [] if requires is None else deepcopy(requires)
    if not isinstance(requires, list):
        raise ValueError("requires must list logical artifact references")
    for reference in requires:
        fields = {"stage", "job_id", "artifact_kind", "slot", "task"}
        if (not isinstance(reference, dict) or set(reference) - fields
                or not {"stage", "job_id", "artifact_kind"}.issubset(reference)):
            raise ValueError("Malformed logical artifact reference")
        if reference["stage"] not in STAGE_OUTPUT_ROOTS or reference["artifact_kind"] not in ARTIFACT_KINDS:
            raise ValueError("Unknown referenced stage or artifact kind")
        require_id(reference["job_id"])
        if "slot" in reference:
            require_id(reference["slot"])
        if "task" in reference and reference["task"] not in TASKS:
            raise ValueError("Unknown referenced test task")
    input_specs = [] if input_specs is None else deepcopy(input_specs)
    if not isinstance(input_specs, list):
        raise ValueError("input_specs must be a list")
    roles = set()
    for spec in input_specs:
        fields = {"role", "kind", "path", "source", "metadata"}
        if not isinstance(spec, dict) or set(spec) != fields:
            raise ValueError("Malformed required-input specification")
        require_id(spec["role"])
        if spec["role"] in roles:
            raise ValueError("Duplicate required-input role")
        roles.add(spec["role"])
        if spec["kind"] not in ARTIFACT_KINDS or not isinstance(spec["path"], str) or not Path(spec["path"]).is_absolute():
            raise ValueError("Required input needs a supported artifact kind and absolute path")
        if not isinstance(spec["source"], str) or not spec["source"].strip() or not isinstance(spec["metadata"], dict):
            raise ValueError("Required input needs source text and metadata constraints")
    lineage = {} if lineage is None else deepcopy(lineage)
    if not isinstance(lineage, dict) or (output is not None and not isinstance(output, dict)):
        raise ValueError("lineage and output must be JSON objects")
    fingerprint({"lineage": lineage, "output": output, "input_specs": input_specs})
    return depends_on, requires, input_specs, lineage, deepcopy(output)


def _normalize_inputs(inputs):
    if not isinstance(inputs, dict):
        raise ValueError("inputs must map semantic roles to artifact manifests")
    normalized = {}
    for role, manifest in inputs.items():
        require_id(role)
        normalized[role] = deepcopy(check_manifest(manifest))
    return normalized


def _manifest_matches(reference, manifest):
    producer = manifest.get("producer")
    if (manifest["kind"] != reference["artifact_kind"] or not isinstance(producer, dict)
            or producer.get("stage") != reference["stage"]
            or producer.get("job_id") != reference["job_id"]):
        return False
    metadata = manifest.get("metadata", {})
    if "slot" in reference and metadata.get("slot") != reference["slot"]:
        return False
    if "task" in reference and metadata.get("task") != reference["task"]:
        return False
    return True


def _validate_input_specs(specs, inputs):
    for spec in specs:
        manifest = inputs.get(spec["role"])
        expected_path = str(Path(spec["path"]).expanduser().absolute())
        if (manifest is None or manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != expected_path
                or manifest["external_source"] != spec["source"]
                or any(manifest["metadata"].get(key) != value for key, value in spec["metadata"].items())):
            raise ValueError("Required external input is missing or mismatched: {}".format(spec["role"]))


def _validate_required_inputs(requires, inputs, experiment_id):
    for reference in requires:
        if not any(_manifest_matches(reference, manifest)
                   and manifest["producer"].get("experiment_id") == experiment_id
                   for manifest in inputs.values()):
            raise ValueError("Required artifact is not explicitly bound: {}".format(reference))


def _training_policy(stage, task, config_fingerprint, inputs, requires, resume):
    if type(resume) is not bool:
        raise ValueError("resume_training_state must be a boolean")
    if stage != "tune":
        if resume:
            raise ValueError("Training-state restore applies only to tune jobs")
        return None
    adapters = [item for item in inputs.values() if item["kind"] == "adapter"]
    checkpoints = [item for item in inputs.values() if item["kind"] == "checkpoint"]
    if len(adapters) > 1:
        raise ValueError("Each training job can inherit only one adapter")
    if resume:
        if len(checkpoints) != 1:
            raise ValueError("Full resume requires one explicit checkpoint")
        # An optional parent adapter records lineage only; the checkpoint supplies all weights/state.
        metadata = checkpoints[0]["metadata"]
        if task is None or metadata.get("training_task") != task:
            raise ValueError("Full resume requires a checkpoint from this same training task")
        if metadata.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Full resume requires the original effective job configuration")
        return "restore_full_checkpoint"
    if checkpoints:
        raise ValueError("Weight-only continuation accepts an adapter, not a full checkpoint")
    if adapters or any(reference["artifact_kind"] == "adapter" for reference in requires):
        return "continue_existing_adapter_reset_optimizer_scheduler_epoch_global_step"
    return "base_model_fresh_optimizer_scheduler_epoch_global_step"


def create_experiment(snapshot, experiment_id=None):
    """Freeze configuration and create an experiment; launch no workloads."""
    validate_snapshot(snapshot)
    experiment = snapshot["config"]["experiment"]
    identifier = require_id(experiment_id if experiment_id is not None else new_id(experiment["name"]))
    directory = Path(experiment["records_root"]) / identifier
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "stage-runs").mkdir()
    now = utc_now()
    with record_lock(directory):
        write_json_atomic(directory / "config.snapshot.json", snapshot)
        write_json_atomic(directory / "experiment.json", {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "kind": "experiment", "experiment_id": identifier,
            "name": experiment["name"], "setting": experiment["setting"],
            "status": "active", "created_at": now, "updated_at": now,
            "config_snapshot": "config.snapshot.json",
            "config_fingerprint": snapshot["config_fingerprint"], "stage_runs": [],
        })
    return directory


def load_experiment(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    _, record = _read_record(directory, "experiment.json")
    return record, _load_snapshot(directory, record)


def create_stage_run(experiment_dir, stage, cli_override=None, run_id=None, stage_plan=None):
    """Create a high-level run from the saved experiment, without rereading YAML."""
    experiment_dir = Path(experiment_dir).expanduser().resolve(strict=True)
    experiment, snapshot = load_experiment(experiment_dir)
    resolved = resolve_run_config(snapshot, stage, cli_override)
    identifier = require_id(run_id if run_id is not None else new_id(stage))
    if stage_plan is not None and (
        not isinstance(stage_plan, dict) or stage_plan.get("stage") != stage
        or stage_plan.get("config_fingerprint") != resolved["config_fingerprint"]
    ):
        raise ValueError("Stage plan does not match the effective run configuration")
    directory = experiment_dir / "stage-runs" / stage / identifier
    now = utc_now()
    record = {
        "record_schema_version": RECORD_SCHEMA_VERSION, "kind": "stage_run",
        "run_id": identifier, "stage": stage, "experiment_id": experiment["experiment_id"],
        "status": "planned", "created_at": now, "updated_at": now,
        "config_snapshot": "config.snapshot.json", "config_fingerprint": resolved["config_fingerprint"],
        "plan": "plan.json" if stage_plan is not None else None,
        "plan_fingerprint": fingerprint(stage_plan) if stage_plan is not None else None,
        "jobs": [],
        "output_root": str(Path(resolved["config"]["experiment"][STAGE_OUTPUT_ROOTS[stage]])
                           / experiment["experiment_id"] / identifier),
    }
    with record_lock(experiment_dir):
        parent_path, parent = _read_record(experiment_dir, "experiment.json")
        if parent["status"] != "active":
            raise ValueError("Cannot add a run to a terminal experiment")
        if any(item["stage"] == stage for item in parent["stage_runs"]):
            raise ValueError("This experiment already has a high-level run for {}".format(stage))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "jobs").mkdir()
        write_json_atomic(directory / "config.snapshot.json", resolved)
        if stage_plan is not None:
            write_json_atomic(directory / "plan.json", stage_plan)
        write_json_atomic(directory / "run.json", record)
        parent["stage_runs"].append({"stage": stage, "run_id": identifier,
                                     "path": directory.relative_to(experiment_dir).as_posix()})
        parent["updated_at"] = utc_now()
        write_json_atomic(parent_path, parent)
    return directory


def _load_planned_jobs(directory, record):
    """Read the frozen plan and index its jobs, if this run has a plan."""
    if record["plan"] is None:
        return None
    if record["plan"] != "plan.json":
        raise ValueError("Stage-run plan fingerprint mismatch")
    plan = read_json(Path(directory) / "plan.json")
    if fingerprint(plan) != record["plan_fingerprint"]:
        raise ValueError("Stage-run plan fingerprint mismatch")
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list):
        raise ValueError("Stage-run plan must contain a jobs list")
    planned_jobs = {}
    for job in plan["jobs"]:
        if not isinstance(job, dict):
            raise ValueError("Every planned job must be an object")
        identifier = require_id(job.get("job_id"))
        if identifier in planned_jobs:
            raise ValueError("Duplicate job ID in frozen plan: {}".format(identifier))
        planned_jobs[identifier] = job
    return planned_jobs


def load_stage_run(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    _, record = _read_record(directory, "run.json")
    snapshot = _load_snapshot(directory, record)
    _load_planned_jobs(directory, record)
    return record, snapshot


def create_job(stage_run_dir, job_id, *, job_override=None, cli_override=None, inputs=None,
               resume_training_state=False, task=None, tasks=None, depends_on=None,
               requires=None, input_specs=None, lineage=None, output=None):
    """Record a low-level job, including logical dependencies and explicit bound inputs."""
    stage_run_dir = Path(stage_run_dir).expanduser().resolve(strict=True)
    run, snapshot = load_stage_run(stage_run_dir)
    identifier = require_id(job_id)
    stage = run["stage"]
    depends_on, requires, input_specs, lineage, output = _job_fields(
        task, tasks, depends_on, requires, input_specs, lineage, output)
    if identifier in depends_on:
        raise ValueError("A job cannot depend on itself")
    if output is not None:
        if (set(output) - {"artifact_kind", "slot", "tasks"}
                or output.get("artifact_kind") != PRIMARY_ARTIFACTS[stage]):
            raise ValueError("Output must describe the primary artifact of this stage")
        require_id(output.get("slot"))
        if stage == "infer" and (tasks is None or output.get("tasks") != tasks):
            raise ValueError("Prediction output must list every test task of its job")
        if stage != "infer" and "tasks" in output:
            raise ValueError("Only prediction outputs use per-task artifact lists")
    resolved = resolve_job_config(snapshot, stage, identifier, job_override, cli_override)
    normalized = _normalize_inputs({} if inputs is None else inputs)
    policy = _training_policy(stage, task, resolved["config_fingerprint"], normalized, requires, resume_training_state)
    directory = stage_run_dir / "jobs" / identifier
    now = utc_now()
    record = {
        "record_schema_version": RECORD_SCHEMA_VERSION, "kind": "job", "job_id": identifier,
        "stage": stage, "run_id": run["run_id"], "experiment_id": run["experiment_id"],
        "status": "planned", "created_at": now, "updated_at": now,
        "config_snapshot": "config.snapshot.json", "config_fingerprint": resolved["config_fingerprint"],
        "task": task, "tasks": deepcopy(tasks), "depends_on": depends_on, "requires": requires,
        "input_specs": input_specs, "lineage": lineage, "inputs": normalized, "output_spec": output,
        "output_dir": str(Path(resolved["config"]["experiment"][STAGE_OUTPUT_ROOTS[stage]])
                          / run["experiment_id"] / run["run_id"] / identifier),
        "training_state_policy": policy, "resume_training_state": resume_training_state,
        "artifacts": [],
    }
    with record_lock(stage_run_dir):
        parent_path, parent = _read_record(stage_run_dir, "run.json")
        if parent["status"] not in ("planned", "running"):
            raise ValueError("Cannot add a job to a terminal stage run")
        if any(item["job_id"] == identifier for item in parent["jobs"]):
            raise ValueError("Job is already registered: {}".format(identifier))
        planned_jobs = _load_planned_jobs(stage_run_dir, parent)
        if planned_jobs is not None:
            if identifier not in planned_jobs:
                raise ValueError("Job is not in the frozen stage-run plan: {}".format(identifier))
            planned = planned_jobs[identifier]
            defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                        "input_specs": [], "lineage": {}, "output": None}
            fields = {"task": task, "tasks": tasks, "depends_on": depends_on, "requires": requires,
                      "input_specs": input_specs, "lineage": lineage, "output": output}
            mismatched = [name for name, value in fields.items()
                          if value != planned.get(name, defaults[name])]
            if mismatched:
                raise ValueError("Job {} differs from the frozen plan in fields: {}".format(
                    identifier, ", ".join(mismatched)))
        if set(depends_on) - {item["job_id"] for item in parent["jobs"]}:
            raise ValueError("Local job dependencies must already exist in this stage run")
        directory.mkdir(exist_ok=False)
        (directory / "artifacts").mkdir()
        write_json_atomic(directory / "config.snapshot.json", resolved)
        write_json_atomic(directory / "job.json", record)
        parent["jobs"].append({"job_id": identifier, "path": directory.relative_to(stage_run_dir).as_posix()})
        parent["updated_at"] = utc_now()
        write_json_atomic(parent_path, parent)
    return directory


def load_job(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    _, record = _read_record(directory, "job.json")
    return record, _load_snapshot(directory, record)


def bind_inputs(job_dir, inputs, *, resume_training_state=None):
    """Bind explicit manifests before execution; never discover a latest checkpoint."""
    job_dir = Path(job_dir).expanduser().resolve(strict=True)
    normalized = _normalize_inputs(inputs)
    with record_lock(job_dir):
        record_path, record = _read_record(job_dir, "job.json")
        if record["status"] != "planned":
            raise ValueError("Inputs can be bound only before a job starts")
        if record["inputs"]:
            raise ValueError("Inputs are already bound; create a new job for a different binding")
        _validate_input_specs(record["input_specs"], normalized)
        _validate_required_inputs(record["requires"], normalized, record["experiment_id"])
        resume = record["resume_training_state"] if resume_training_state is None else resume_training_state
        policy = _training_policy(record["stage"], record["task"], record["config_fingerprint"],
                                  normalized, record["requires"], resume)
        record["inputs"] = normalized
        record["resume_training_state"] = resume
        record["training_state_policy"] = policy
        record["updated_at"] = utc_now()
        write_json_atomic(record_path, record)
    return record


def register_artifact(job_dir, role, manifest):
    """Attach a job-produced artifact manifest without copying its payload."""
    job_dir = Path(job_dir).expanduser().resolve(strict=True)
    role = require_id(role)
    manifest = deepcopy(check_manifest(manifest, verify_contents=True))
    artifact_path = job_dir / "artifacts" / (role + ".json")
    with record_lock(job_dir):
        record_path, record = _read_record(job_dir, "job.json")
        if record["status"] not in ("running", "succeeded"):
            raise ValueError("Artifacts can be registered only for a running or succeeded job")
        producer = {key: record[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
        if manifest["producer"] != producer:
            raise ValueError("Output manifest must identify this exact producing job")
        if manifest["kind"] not in STAGE_ARTIFACTS[record["stage"]]:
            raise ValueError("Artifact kind is not produced by this stage")
        expected = record["output_spec"] or {}
        if manifest["kind"] == expected.get("artifact_kind"):
            if manifest["metadata"].get("slot") != expected.get("slot"):
                raise ValueError("Primary artifact slot does not match the planned output")
            if manifest["kind"] == "prediction" and manifest["metadata"].get("task") not in expected["tasks"]:
                raise ValueError("Prediction must identify one planned test task")
        if manifest["kind"] == expected.get("artifact_kind"):
            for item in record["artifacts"]:
                if item["kind"] != manifest["kind"]:
                    continue
                previous = check_manifest(read_json(job_dir / item["path"]))
                if (manifest["kind"] != "prediction"
                        or previous["metadata"].get("task") == manifest["metadata"].get("task")):
                    raise ValueError("Primary output is already registered for this slot/task")
        if any(item["role"] == role for item in record["artifacts"]) or artifact_path.exists():
            raise ValueError("Artifact role already exists: {}".format(role))
        write_json_atomic(artifact_path, manifest)
        record["artifacts"].append({"role": role, "kind": manifest["kind"],
                                    "path": "artifacts/" + artifact_path.name,
                                    "fingerprint": manifest["fingerprint"]})
        record["updated_at"] = utc_now()
        write_json_atomic(record_path, record)
    return artifact_path


def record_status(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    for filename in ("experiment.json", "run.json", "job.json"):
        if (directory / filename).is_file():
            _, record = _read_record(directory, filename)
            return record
    raise ValueError("No recognized record in {}".format(directory))


def update_status(directory, new_status, message=None):
    """Update lifecycle metadata only; this function never starts or stops workloads."""
    directory = Path(directory).expanduser().resolve(strict=True)
    if not isinstance(new_status, str) or (message is not None and
                                           (not isinstance(message, str) or not message.strip())):
        raise ValueError("Status and optional message must be non-empty text")
    with record_lock(directory):
        record = record_status(directory)
        old_status = record["status"]
        if new_status not in TRANSITIONS.get(old_status, set()):
            raise ValueError("Invalid status transition: {} -> {}".format(old_status, new_status))
        if new_status == "running" and record["kind"] == "job":
            _validate_input_specs(record["input_specs"], record["inputs"])
            _validate_required_inputs(record["requires"], record["inputs"], record["experiment_id"])
            for dependency in record["depends_on"]:
                sibling = record_status(directory.parent / dependency)
                if sibling["status"] != "succeeded":
                    raise ValueError("Local job dependencies must succeed before this job starts")
        if new_status == "succeeded":
            if record["kind"] == "job":
                expected = record["output_spec"] or {}
                outputs = []
                for item in record["artifacts"]:
                    manifest = check_manifest(read_json(directory / item["path"]))
                    if manifest["fingerprint"] != item["fingerprint"]:
                        raise ValueError("Registered artifact manifest has changed")
                    if (manifest["kind"] == expected.get("artifact_kind")
                            and manifest["metadata"].get("slot") == expected.get("slot")):
                        outputs.append(manifest)
                if expected and not outputs:
                    raise ValueError("Job cannot succeed before its primary output is registered")
                if expected.get("artifact_kind") == "prediction":
                    completed = {item["metadata"].get("task") for item in outputs}
                    if completed != set(expected["tasks"]):
                        raise ValueError("Every planned test task requires its own prediction artifact")
            elif record["kind"] in ("stage_run", "experiment"):
                children = record["jobs"] if record["kind"] == "stage_run" else record["stage_runs"]
                planned_jobs = _load_planned_jobs(directory, record) if record["kind"] == "stage_run" else None
                if planned_jobs is not None:
                    registered_ids = [child["job_id"] for child in children]
                    if len(set(registered_ids)) != len(registered_ids):
                        raise ValueError("Stage run contains duplicate registered job IDs")
                    missing = set(planned_jobs) - set(registered_ids)
                    unexpected = set(registered_ids) - set(planned_jobs)
                    if missing or unexpected:
                        raise ValueError("Stage-run jobs do not match the frozen plan (missing: {}; unexpected: {})".format(
                            ", ".join(sorted(missing)) or "none",
                            ", ".join(sorted(unexpected)) or "none"))
                if not children:
                    raise ValueError("Cannot succeed with no child records")
                for child in children:
                    child_dir = directory / child["path"] if record["kind"] == "stage_run" else directory / "stage-runs" / child["stage"] / child["run_id"]
                    child_record = record_status(child_dir)
                    if planned_jobs is not None and (child_record["kind"] != "job"
                            or child_record["job_id"] != child["job_id"]):
                        raise ValueError("Registered job record does not match its job ID: {}".format(child["job_id"]))
                    if child_record["status"] != "succeeded":
                        raise ValueError("All child records must succeed first")
        filename = {"experiment": "experiment.json", "stage_run": "run.json", "job": "job.json"}[record["kind"]]
        record["status"] = new_status
        record["updated_at"] = utc_now()
        if message is not None:
            record["status_message"] = message
        write_json_atomic(directory / filename, record)
    return record

