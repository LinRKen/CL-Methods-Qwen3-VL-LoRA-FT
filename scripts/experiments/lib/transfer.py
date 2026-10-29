"""Validate and freeze reuse of a completed transfer-round1 tune run."""

from pathlib import Path

from .artifacts import check_manifest
from .common import TASKS, fingerprint, read_json
from .config import resolve_job_config
from .records import load_experiment, load_job, load_stage_run
from .tune_orchestrator import _successful_adapter


def _pipeline_helpers():
    # Import lazily because pipeline imports this module while it is loading.
    if __package__ == "lib":
        from pipeline import _tune_jobs, build_stage_plan
    else:
        from ..pipeline import _tune_jobs, build_stage_plan
    return _tune_jobs, build_stage_plan


def _same_path(left, right):
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _source_experiment(snapshot, current_experiment_id):
    experiment = snapshot["config"]["experiment"]
    source_id = experiment.get("transfer_source_experiment_id")
    if source_id is None:
        if "transfer_source" in snapshot["layers"]:
            raise ValueError("A transfer source binding exists without a source experiment ID")
        return None
    if current_experiment_id is not None and source_id == current_experiment_id:
        raise ValueError("A transfer experiment cannot reuse itself as its source")
    source_root = Path(experiment["records_root"]).resolve()
    source_dir = source_root / source_id
    source_record, source_snapshot = load_experiment(source_dir)
    if source_record["experiment_id"] != source_id:
        raise ValueError("Transfer source experiment identity does not match its directory")
    if source_record["status"] not in ("active", "succeeded"):
        raise ValueError("Transfer source experiment must be active or succeeded")
    source_experiment = source_snapshot["config"]["experiment"]
    if source_experiment["setting"] != "transfer_round1":
        raise ValueError("Transfer source experiment must use setting transfer_round1")
    if source_experiment.get("transfer_source_experiment_id") is not None:
        raise ValueError("A transfer-round1 source cannot itself reuse another experiment")
    if not _same_path(source_experiment["records_root"], experiment["records_root"]):
        raise ValueError("Transfer source and current experiment must use the same records_root")
    for key in ("base_model", "processor_path", "dataset_root", "tasks", "task_order"):
        if source_experiment[key] != experiment[key]:
            raise ValueError("Transfer source and current experiment differ in experiment.{}".format(key))
    return source_id, source_dir, source_record, source_snapshot


def _source_tune(source_id, source_dir, source_record, source_snapshot):
    registrations = [item for item in source_record["stage_runs"] if item["stage"] == "tune"]
    if len(registrations) != 1:
        raise ValueError("Transfer source must have exactly one registered tune run")
    registration = registrations[0]
    tune_dir = source_dir / "stage-runs" / "tune" / registration["run_id"]
    if registration.get("path") != tune_dir.relative_to(source_dir).as_posix():
        raise ValueError("Transfer source tune registration path is inconsistent")
    tune_run, tune_snapshot = load_stage_run(tune_dir)
    if (tune_run["stage"] != "tune" or tune_run["run_id"] != registration["run_id"]
            or tune_run["experiment_id"] != source_id or tune_run["status"] != "succeeded"
            or tune_snapshot["config"]["experiment"] != source_snapshot["config"]["experiment"]
            or tune_snapshot["layers"].get("parent_snapshot") != source_snapshot["snapshot_fingerprint"]):
        raise ValueError("Transfer source tune run is not the exact succeeded child of its experiment")
    return tune_dir, tune_run, tune_snapshot


def _checked_manifest(manifest, *, verify_contents, cache):
    """Verify a manifest, hashing each distinct source path at most once per audit."""
    key = (manifest.get("kind"), manifest.get("path"), manifest.get("fingerprint"))
    if verify_contents and key in cache:
        return check_manifest(manifest, verify_contents=False)
    result = check_manifest(manifest, verify_contents=verify_contents)
    if verify_contents:
        cache.add(key)
    return result


def _validate_external_inputs(job, *, verify_contents, manifest_cache):
    specs = {item["role"]: item for item in job["input_specs"]}
    for role, spec in specs.items():
        manifest = job["inputs"].get(role)
        if manifest is None:
            raise ValueError("Source tune input is not bound: {}:{}".format(job["job_id"], role))
        _checked_manifest(manifest, verify_contents=verify_contents, cache=manifest_cache)
        if (manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != str(Path(spec["path"]).expanduser().absolute())
                or manifest["path"] != str(Path(spec["path"]).expanduser().resolve(strict=True))
                or manifest["external_source"] != spec["source"]
                or manifest["metadata"] != spec["metadata"]):
            raise ValueError("Source tune input differs from its frozen specification: {}:{}".format(
                job["job_id"], role))
    return specs


def _source_adapter(job_dir, job, *, verify_contents, adapter_cache):
    cached = adapter_cache.get(job["job_id"])
    if cached is not None and (not verify_contents or cached["verified_contents"]):
        return cached["adapter"]
    adapter = _successful_adapter(job_dir, job, verify_contents=verify_contents)
    adapter_cache[job["job_id"]] = {
        "adapter": adapter,
        "verified_contents": verify_contents or (cached is not None and cached["verified_contents"]),
    }
    return adapter


def _validate_parent(job, tune_dir, tune_run, *, verify_contents, manifest_cache, adapter_cache):
    if job["lineage"].get("round") != 2:
        if any(item["kind"] == "adapter" for item in job["inputs"].values()):
            raise ValueError("A first-round source tune job cannot have an adapter input")
        return
    expected_parent = "tune-round1-{}".format(job["lineage"].get("source_task"))
    references = job["requires"]
    if (len(references) != 1 or references[0].get("stage") != "tune"
            or references[0].get("artifact_kind") != "adapter"
            or references[0].get("job_id") != expected_parent
            or references[0].get("slot") != "round1-{}".format(job["lineage"].get("source_task"))):
        raise ValueError("Second-round source job has an invalid parent reference: " + job["job_id"])
    parent_dir = tune_dir / "jobs" / expected_parent
    parent_job, _ = load_job(parent_dir)
    if (parent_job["status"] != "succeeded" or parent_job["run_id"] != tune_run["run_id"]
            or parent_job["experiment_id"] != tune_run["experiment_id"]):
        raise ValueError("Second-round source job parent has not succeeded: " + job["job_id"])
    parent = _source_adapter(parent_dir, parent_job, verify_contents=verify_contents,
                             adapter_cache=adapter_cache)
    adapters = [item for item in job["inputs"].values() if item["kind"] == "adapter"]
    if len(adapters) != 1:
        raise ValueError("Second-round source job must bind one parent adapter: " + job["job_id"])
    bound = _checked_manifest(adapters[0], verify_contents=verify_contents, cache=manifest_cache)
    registered = _checked_manifest(read_json(parent["manifest"]), verify_contents=verify_contents,
                                   cache=manifest_cache)
    if bound != registered:
        raise ValueError("Second-round source job is bound to a different parent adapter: " + job["job_id"])


def _validate_source_job(job_id, planned, tune_dir, tune_run, tune_snapshot, current_snapshot,
                         *, verify_contents, manifest_cache, adapter_cache):
    job_dir = tune_dir / "jobs" / job_id
    job, job_snapshot = load_job(job_dir)
    if (job["job_id"] != job_id or job["stage"] != "tune"
            or job["run_id"] != tune_run["run_id"]
            or job["experiment_id"] != tune_run["experiment_id"]
            or job_snapshot["layers"].get("parent_snapshot") != tune_snapshot["snapshot_fingerprint"]
            or job["status"] != "succeeded"):
        raise ValueError("Transfer source tune job is not a succeeded frozen job: " + job_id)
    resolved_source = resolve_job_config(
        tune_snapshot, "tune", job_id,
        job_snapshot["layers"].get("job"), job_snapshot["layers"].get("job_cli"))
    if resolved_source["config_fingerprint"] != job["config_fingerprint"]:
        raise ValueError("Transfer source job configuration cannot be reproduced: " + job_id)
    defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                "input_specs": [], "lineage": {}, "output": None}
    for field, default in defaults.items():
        actual = job["output_spec" if field == "output" else field]
        expected = planned.get("output" if field == "output" else field, default)
        if actual != expected:
            raise ValueError("Transfer source job differs from its frozen plan: " + job_id)
    resolved_current = resolve_job_config(current_snapshot, "tune", job_id, allow_disabled=True)
    if resolved_source["config"]["tune"]["params"] != resolved_current["config"]["tune"]["params"]:
        raise ValueError("Transfer source and current tune parameters differ: " + job_id)
    expected_output = (Path(tune_snapshot["config"]["experiment"]["checkpoints_root"])
                       / tune_run["experiment_id"] / tune_run["run_id"] / job_id)
    if Path(job["output_dir"]) != expected_output:
        raise ValueError("Transfer source adapter output directory is inconsistent: " + job_id)
    _validate_external_inputs(job, verify_contents=verify_contents, manifest_cache=manifest_cache)
    _validate_parent(job, tune_dir, tune_run, verify_contents=verify_contents,
                     manifest_cache=manifest_cache, adapter_cache=adapter_cache)
    adapter = _source_adapter(job_dir, job, verify_contents=verify_contents,
                              adapter_cache=adapter_cache)
    return {
        "directory": job_dir,
        "job": job,
        "snapshot": job_snapshot,
        "adapter": adapter,
        "adapter_manifest": read_json(adapter["manifest"]),
    }


def validate_transfer_source(snapshot, experiment_id=None, *, audit_adapters=True,
                             require_frozen=False, adapter_job_id=None):
    """Validate the configured round1 source and return its frozen adapter binding."""
    source = _source_experiment(snapshot, experiment_id)
    if source is None:
        if require_frozen:
            raise ValueError("A frozen transfer source binding is required")
        return None
    source_id, source_dir, source_record, source_snapshot = source
    tune_dir, tune_run, tune_snapshot = _source_tune(
        source_id, source_dir, source_record, source_snapshot)
    _tune_jobs, build_stage_plan = _pipeline_helpers()
    canonical_plan = build_stage_plan(tune_snapshot, "tune")
    plan = read_json(tune_dir / "plan.json")
    if plan != canonical_plan or tune_run["plan_fingerprint"] != fingerprint(plan):
        raise ValueError("Transfer source tune plan differs from its frozen configuration")
    planned = {item["job_id"]: item for item in canonical_plan["jobs"]}
    current_jobs = _tune_jobs(snapshot["config"])
    current_by_id = {item["job_id"]: item for item in current_jobs}
    expected_ids = set(planned)
    if (len(planned) != 36 or set(current_by_id) != expected_ids
            or set(snapshot["config"]["tune"]["jobs"]) - expected_ids):
        raise ValueError("Transfer source and current experiment must use the same 36 tune jobs")
    registered = {item["job_id"]: item for item in tune_run["jobs"]}
    if set(registered) != expected_ids or len(tune_run["jobs"]) != len(expected_ids):
        raise ValueError("Transfer source tune registration does not contain exactly 36 jobs")

    verify_contents = bool(audit_adapters)
    if adapter_job_id is not None and adapter_job_id not in planned:
        raise ValueError("Requested transfer adapter job is not in the source plan: " + adapter_job_id)
    manifest_cache = set()
    adapter_cache = {}
    jobs = {}
    for job_id in planned:
        if registered[job_id].get("path") != "jobs/" + job_id:
            raise ValueError("Transfer source job registration path is inconsistent: " + job_id)
        jobs[job_id] = _validate_source_job(
            job_id, planned[job_id], tune_dir, tune_run, tune_snapshot, snapshot,
            verify_contents=verify_contents or job_id == adapter_job_id,
            manifest_cache=manifest_cache, adapter_cache=adapter_cache)

    binding = {
        "experiment_id": source_id,
        "experiment_snapshot_fingerprint": source_snapshot["snapshot_fingerprint"],
        "tune_run_id": tune_run["run_id"],
        "tune_snapshot_fingerprint": tune_snapshot["snapshot_fingerprint"],
        "tune_plan_fingerprint": tune_run["plan_fingerprint"],
        "jobs": {
            job_id: {
                "config_fingerprint": info["job"]["config_fingerprint"],
                "record_fingerprint": fingerprint(info["job"]),
                "adapter_fingerprint": info["adapter"]["fingerprint"],
                "adapter_producer": info["adapter"]["producer"],
                "adapter_path": info["adapter"]["path"],
            }
            for job_id, info in jobs.items()
        },
    }
    if require_frozen and snapshot["layers"].get("transfer_source") != binding:
        raise ValueError("Frozen transfer source binding no longer matches the source records")
    return {
        "experiment_id": source_id,
        "source_dir": source_dir,
        "experiment": source_record,
        "experiment_snapshot": source_snapshot,
        "tune_dir": tune_dir,
        "tune_run": tune_run,
        "tune_snapshot": tune_snapshot,
        "jobs": jobs,
        "binding": binding,
    }


def validate_transfer_infer_job(snapshot, job, source_context, adapter_manifest=None):
    """Validate one round2 infer job against the frozen external tune source."""
    experiment = snapshot["config"]["experiment"]
    source_id = source_context["experiment_id"]
    if (experiment["setting"] != "transfer_round2"
            or experiment.get("transfer_source_experiment_id") != source_id):
        raise ValueError("Infer job is not attached to the configured transfer source")
    source_task = job["lineage"].get("source_task")
    target_task = job["lineage"].get("second_round_task")
    if not isinstance(source_task, str) or not isinstance(target_task, str):
        raise ValueError("Infer job lacks a transfer source/target task pair: " + job["job_id"])
    slot = "round2-{}-to-{}".format(source_task, target_task)
    source_job_id = "tune-round2-{}-to-{}".format(source_task, target_task)
    if (job["job_id"] != "infer-" + slot
            or job["tasks"] != [target_task]
            or job["lineage"].get("adapter_ref") != source_job_id
            or job["lineage"].get("adapter_source_experiment_id") != source_id
            or job["lineage"].get("test_tasks") != [target_task]
            or job["output_spec"] != {"artifact_kind": "prediction", "slot": slot, "tasks": [target_task]}
            or source_task not in TASKS or target_task not in TASKS or source_task == target_task):
        raise ValueError("Infer job does not describe the expected round2 source/target pair: " + job["job_id"])
    expected_reference = {
        "experiment_id": source_id, "stage": "tune", "job_id": source_job_id,
        "artifact_kind": "adapter", "slot": slot,
    }
    if job["requires"] != [expected_reference]:
        raise ValueError("Infer job has an unexpected external adapter reference: " + job["job_id"])
    source_info = source_context["jobs"].get(source_job_id)
    if source_info is None:
        raise ValueError("Transfer source has no matching round2 adapter: " + source_job_id)
    source_job = source_info["job"]
    if (source_job["task"] != target_task
            or source_job["output_spec"] != {"artifact_kind": "adapter", "slot": slot}
            or source_job["lineage"].get("round") != 2
            or source_job["lineage"].get("source_task") != source_task
            or source_job["lineage"].get("train_task") != target_task):
        raise ValueError("Transfer source adapter job does not match the infer pair: " + job["job_id"])
    if adapter_manifest is not None:
        check_manifest(adapter_manifest, verify_contents=True)
        expected = source_info["adapter"]
        if (adapter_manifest["fingerprint"] != expected["fingerprint"]
                or adapter_manifest["path"] != expected["path"]
                or adapter_manifest["producer"] != expected["producer"]
                or adapter_manifest["metadata"].get("slot") != slot):
            raise ValueError("Infer job is bound to a different transfer adapter: " + job["job_id"])
    return source_info
