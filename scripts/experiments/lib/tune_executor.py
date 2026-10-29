"""Execute one frozen tune job. Importing this module never initializes CUDA."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .artifacts import check_manifest, make_manifest
from .common import fingerprint, read_json, sha256_file, utc_now, write_json_atomic
from .config import resolve_job_config
from .records import bind_inputs, load_experiment, load_job, load_stage_run, register_artifact, update_status

FRESH = "base_model_fresh_optimizer_scheduler_epoch_global_step"
CONTINUE = "continue_existing_adapter_reset_optimizer_scheduler_epoch_global_step"
RESTORE = "restore_full_checkpoint"


@contextmanager
def _exclusive(path, *, blocking=False):
    """Coordinate concurrent job launches and shared local GPU/port resources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as error:
            raise ValueError("Another executor holds {}".format(path)) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _stage_slot(run_dir, capacity):
    """Apply the parent run's concurrency limit to independently launched jobs."""
    directory = run_dir / ".tune-slots"
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(capacity):
            handle = (directory / "{}.lock".format(index)).open("a", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            except BaseException:
                handle.close()
                raise
            try:
                yield index
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.2)


def _start_stage(run_dir):
    """Accept another executor having started the shared parent run first."""
    current, _ = load_stage_run(run_dir)
    if current["status"] == "planned":
        try:
            update_status(run_dir, "running", "First low-level tune job started")
        except ValueError:
            current, _ = load_stage_run(run_dir)
            if current["status"] != "running":
                raise
    elif current["status"] != "running":
        raise ValueError("The parent tune run is no longer active")


def _context(job_dir):
    job, snapshot = load_job(job_dir)
    run_dir = job_dir.parent.parent
    experiment_dir = run_dir.parents[2]
    run, run_snapshot = load_stage_run(run_dir)
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    if (job["stage"] != "tune" or run["stage"] != "tune"
            or job_dir != run_dir / "jobs" / job["job_id"]
            or run_dir != experiment_dir / "stage-runs" / "tune" / run["run_id"]
            or experiment_dir.name != experiment["experiment_id"]
            or job["experiment_id"] != experiment["experiment_id"]
            or run["experiment_id"] != experiment["experiment_id"]
            or job["run_id"] != run["run_id"]):
        raise ValueError("Job, stage run and experiment identities disagree")
    if experiment_dir != Path(snapshot["config"]["experiment"]["records_root"]).resolve() / job["experiment_id"]:
        raise ValueError("Record directory disagrees with the frozen records_root")
    if job["status"] != "planned" or job["artifacts"]:
        raise ValueError("Only an unexecuted planned job can be launched")
    if experiment["status"] != "active" or run["status"] not in ("planned", "running"):
        raise ValueError("The owning experiment and stage run must still be active")
    if {"stage": "tune", "run_id": run["run_id"],
            "path": run_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]:
        raise ValueError("Stage run is not registered with its experiment")
    if {"job_id": job["job_id"], "path": "jobs/" + job["job_id"]} not in run["jobs"]:
        raise ValueError("Job is not registered with its stage run")
    if run_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]:
        raise ValueError("Stage run has a different frozen parent snapshot")
    if snapshot["layers"].get("parent_snapshot") != run_snapshot["snapshot_fingerprint"]:
        raise ValueError("Job has a different frozen parent snapshot")
    resolved = resolve_job_config(run_snapshot, "tune", job["job_id"],
                                  snapshot["layers"].get("job"), snapshot["layers"].get("job_cli"))
    if resolved["config_fingerprint"] != job["config_fingerprint"]:
        raise ValueError("Job configuration cannot be reproduced from its frozen parent")
    if snapshot["config"]["experiment"] != run_snapshot["config"]["experiment"]:
        raise ValueError("Job cannot override experiment paths or model/data lineage")
    if run["plan"] != "plan.json":
        raise ValueError("The tune executor requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    if plan.get("stage") != "tune" or plan.get("config_fingerprint") != run["config_fingerprint"]:
        raise ValueError("Frozen plan does not describe this tune run")
    planned = [item for item in plan["jobs"] if item["job_id"] == job["job_id"]]
    if len(planned) != 1:
        raise ValueError("Job must occur exactly once in the frozen plan")
    defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                "input_specs": [], "lineage": {}, "output": None}
    for field, default in defaults.items():
        actual = job["output_spec" if field == "output" else field]
        if actual != planned[0].get(field, default):
            raise ValueError("Job no longer matches planned field: " + field)
    if not job["task"] or job["tasks"] is not None or not job["output_spec"]:
        raise ValueError("A tune job needs one training task and an adapter output slot")
    for dependency in job["depends_on"]:
        sibling, _ = load_job(run_dir / "jobs" / dependency)
        if (sibling["status"] != "succeeded" or sibling["run_id"] != job["run_id"]
                or sibling["experiment_id"] != job["experiment_id"]):
            raise ValueError("A training dependency has not succeeded: " + dependency)
    return job, snapshot, experiment_dir, run


def _registered_producer(manifest, records_root):
    """Resolve an explicitly supplied manifest's exact registered producing job."""
    producer = manifest["producer"]
    if not isinstance(producer, dict) or producer.get("stage") != "tune":
        raise ValueError("Inherited training weights require a registered tune producer")
    source_experiment_dir = Path(records_root) / producer["experiment_id"]
    source_experiment, source_snapshot = load_experiment(source_experiment_dir)
    source_run_dir = source_experiment_dir / "stage-runs" / producer["stage"] / producer["run_id"]
    source_run, _ = load_stage_run(source_run_dir)
    source_dir = source_run_dir / "jobs" / producer["job_id"]
    source_job, _ = load_job(source_dir)
    allowed_statuses = {"succeeded"} if manifest["kind"] == "adapter" else {"succeeded", "failed", "cancelled"}
    if (source_experiment["experiment_id"] != producer["experiment_id"]
            or Path(source_snapshot["config"]["experiment"]["records_root"]).resolve() != Path(records_root).resolve()
            or any(source_run[key] != producer[key] for key in ("experiment_id", "stage", "run_id"))
            or any(source_job[key] != value for key, value in producer.items())
            or source_job["status"] not in allowed_statuses
            or {"stage": producer["stage"], "run_id": producer["run_id"],
                "path": source_run_dir.relative_to(source_experiment_dir).as_posix()} not in source_experiment["stage_runs"]
            or {"job_id": producer["job_id"], "path": "jobs/" + producer["job_id"]} not in source_run["jobs"]):
        raise ValueError("Inherited manifest does not identify an eligible registered producer")
    if not any(item["kind"] == manifest["kind"] and item["fingerprint"] == manifest["fingerprint"]
               and read_json(source_dir / item["path"]) == manifest for item in source_job["artifacts"]):
        raise ValueError("Inherited manifest is not registered by its claimed producer")
    return source_job


def _require_internal_inputs(job, inputs, experiment_dir):
    """Keep business dependencies distinct from an explicit recovery checkpoint."""
    matched = set()
    for reference in job["requires"]:
        if reference["artifact_kind"] != "adapter" or reference["stage"] != "tune":
            raise ValueError("A tune plan can inherit only a tune adapter as a business dependency")
        expected_experiment = reference.get("experiment_id", job["experiment_id"])
        if job["stage"] == "tune" and expected_experiment != job["experiment_id"]:
            raise ValueError("Tune jobs cannot inherit an adapter from another experiment")
        matches = []
        for manifest in inputs.values():
            producer = manifest["producer"] or {}
            if (manifest["kind"] == reference["artifact_kind"]
                    and producer.get("experiment_id") == expected_experiment
                    and producer.get("stage") == reference["stage"]
                    and producer.get("job_id") == reference["job_id"]
                    and all(manifest["metadata"].get(key) == reference[key]
                            for key in ("slot", "task") if key in reference)):
                matches.append(manifest)
        if len(matches) != 1:
            raise ValueError("Bind exactly one manifest for required input: {}".format(reference))
        manifest = matches[0]
        _registered_producer(manifest, experiment_dir.parent)
        matched.add(manifest["fingerprint"])
    adapters = {manifest["fingerprint"] for manifest in inputs.values() if manifest["kind"] == "adapter"}
    if adapters != matched:
        raise ValueError("Every parent adapter must match a frozen training dependency")


def _adapter_config(path, params, base_path):
    path = Path(path)
    config = read_json(path / "adapter_config.json")
    targets = config.get("target_modules")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or any(config.get(key) != params[argument] for key, argument in (
                ("r", "lora_r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")))
            or not isinstance(targets, list) or set(targets) != {"q_proj", "k_proj", "v_proj", "o_proj"}
            or config.get("bias", "none") != "none" or config.get("modules_to_save")
            or any(config.get(key) for key in ("rank_pattern", "alpha_pattern", "use_dora", "use_rslora"))):
        raise ValueError("Adapter is incompatible with the frozen language LoRA configuration")
    if config.get("exclude_modules") != r".*visual[.].*":
        raise ValueError("Adapter must explicitly exclude visual modules")
    saved_base = config.get("base_model_name_or_path")
    if not isinstance(saved_base, str) or Path(saved_base).resolve() != Path(base_path).resolve():
        raise ValueError("Adapter was not created for the frozen base-model path")
    if not any((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Adapter has no nonempty weight payload")


def _checkpoint_state(path, world_size, deepspeed, fp16):
    """Require the Trainer/DeepSpeed state layout; do not unpickle optimizer files."""
    path = Path(path)
    state = read_json(path / "trainer_state.json")
    step = state.get("global_step")
    if type(step) is not int or step <= 0:
        raise ValueError("Checkpoint requires a positive completed global_step")
    names = ["rng_state.pth"] if world_size == 1 else [
        "rng_state_{}.pth".format(rank) for rank in range(world_size)]
    if deepspeed:
        tag = "global_step{}".format(step)
        # DeepSpeed's tag pointer belongs to this explicitly selected checkpoint.
        # It is not a search for the latest Trainer checkpoint.
        if (path / "latest").read_text().strip() != tag:
            raise ValueError("DeepSpeed tag does not match the explicit Trainer checkpoint")
        state_dir = path / tag
        optimizers = list(state_dir.glob("*optim_states.pt"))
        models = list(state_dir.glob("*model_states.pt"))
        if len(optimizers) < world_size or not models:
            raise ValueError("DeepSpeed optimizer/model state shards are incomplete")
        names.extend(str(item.relative_to(path)) for item in optimizers + models)
        ds_config = read_json(deepspeed)
        # With neither a DS optimizer nor a DS scheduler, Trainer creates the
        # HF optimizer/scheduler; Accelerate wraps that scheduler and Trainer
        # saves scheduler.pt separately. DummyOptim/DummyScheduler instead hand
        # the scheduler to DeepSpeed, whose model state contains it.
        if "optimizer" not in ds_config and "scheduler" not in ds_config:
            names.append("scheduler.pt")
    else:
        names.extend(["optimizer.pt", "scheduler.pt"])
        if fp16:
            names.append("scaler.pt")
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in names):
        raise ValueError("Checkpoint optimizer/scheduler/RNG state is incomplete")
    return {"global_step": step, "world_size": world_size,
            "backend": "deepspeed" if deepspeed else "trainer", "state_files": sorted(names)}


def _inputs(job, snapshot, supplied, resume_training_state, experiment_dir, world_size):
    config = snapshot["config"]
    experiment, params = config["experiment"], config["tune"]["params"]
    specs = {spec["role"]: spec for spec in job["input_specs"]}
    annotation_role = "annotation-{}-train".format(job["task"])
    expected = {"base-model": ("base_model", experiment["base_model"]),
                "processor": ("processor", experiment["processor_path"]),
                annotation_role: ("annotation", str(Path(experiment["dataset_root"]) / job["task"] / "train.json"))}
    if set(specs) != set(expected):
        raise ValueError("Tune plan must declare a base model, processor and one train annotation")
    for role, (kind, path) in expected.items():
        if specs[role]["kind"] != kind or specs[role]["path"] != path:
            raise ValueError("Frozen input specification conflicts with configuration: " + role)
    if specs[annotation_role]["metadata"] != {"task": job["task"], "split": "train"}:
        raise ValueError("Annotation task/split must match this training job")
    if job["inputs"]:
        if supplied or (resume_training_state is not None
                        and resume_training_state != job["resume_training_state"]):
            raise ValueError("Bound inputs and restore mode cannot be overridden")
        inputs = deepcopy(job["inputs"])
    else:
        inputs = deepcopy(supplied or {})
        for role, spec in specs.items():
            if role not in inputs:
                inputs[role] = make_manifest(spec["kind"], spec["path"],
                                            source=spec["source"], metadata=spec["metadata"])
    for role, manifest in inputs.items():
        check_manifest(manifest, verify_contents=True)
        if role not in specs and manifest["kind"] not in ("adapter", "checkpoint"):
            raise ValueError("Unexpected tune input role: " + role)
    for role, spec in specs.items():
        manifest = inputs.get(role)
        if (manifest is None or manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != str(Path(spec["path"]).absolute())
                or manifest["external_source"] != spec["source"]
                or any(manifest["metadata"].get(key) != value for key, value in spec["metadata"].items())):
            raise ValueError("Input manifest disagrees with frozen specification: " + role)
    _require_internal_inputs(job, inputs, experiment_dir)
    adapters = [value for value in inputs.values() if value["kind"] == "adapter"]
    checkpoints = [value for value in inputs.values() if value["kind"] == "checkpoint"]
    if len(adapters) > 1 or len(checkpoints) > 1:
        raise ValueError("A tune job can use at most one parent adapter and one explicit checkpoint")
    resume = job["resume_training_state"] if resume_training_state is None else resume_training_state
    if type(resume) is not bool or (checkpoints and not resume) or (resume and not checkpoints):
        raise ValueError("Full restore requires one explicit checkpoint and resume_training_state=true")
    identities = {role: inputs[role]["fingerprint"] for role in expected}
    for manifest in adapters + checkpoints:
        if manifest["producer"] is None:
            raise ValueError("Inherited adapters and checkpoints must come from a registered experiment job")
        _adapter_config(manifest["path"], params, inputs["base-model"]["path"])
        recorded_base = manifest["metadata"].get("base_model_fingerprint")
        if recorded_base != identities["base-model"]:
            raise ValueError("Inherited weights have a different or unrecorded base-model identity")
    policy = RESTORE if resume else CONTINUE if adapters else FRESH
    deepspeed = params["deepspeed"]
    deepspeed_sha256 = sha256_file(deepspeed) if deepspeed else None
    if checkpoints:
        checkpoint = checkpoints[0]
        metadata = checkpoint["metadata"]
        source_job = _registered_producer(checkpoint, experiment["records_root"])
        if (source_job["task"] != job["task"]
                or source_job["config_fingerprint"] != job["config_fingerprint"]
                or source_job["lineage"] != job["lineage"]
                or metadata.get("training_task") != job["task"]
                or metadata.get("config_fingerprint") != job["config_fingerprint"]
                or metadata.get("input_fingerprints") != identities
                or metadata.get("world_size") != world_size
                or metadata.get("deepspeed_sha256") != deepspeed_sha256):
            raise ValueError("Full restore requires the same task/configuration/inputs/world size/backend")
        source_parents = [item for item in source_job["inputs"].values() if item["kind"] == "adapter"]
        if ([item["identity"] for item in source_parents] != [item["identity"] for item in adapters]):
            raise ValueError("Recovery checkpoint and planned job have different parent-adapter weights")
        layout = _checkpoint_state(checkpoint["path"], world_size, deepspeed, params["fp16"])
        if metadata.get("checkpoint_state") != layout:
            raise ValueError("Checkpoint state layout no longer matches its recorded metadata")
    if not job["inputs"]:
        job = bind_inputs(Path(experiment_dir) / "stage-runs" / "tune" / job["run_id"] / "jobs" / job["job_id"],
                          inputs, resume_training_state=resume)
    if job["training_state_policy"] != policy:
        raise ValueError("Bound input policy disagrees with the job training-state policy")
    return job, inputs, identities, deepspeed_sha256


def _environment(config, job, output, group):
    run = config["tune"]["run"]
    env = os.environ.copy()
    for key in list(env):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
            del env[key]
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(map(str, group)), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "WANDB_PROJECT": run["wandb_project"],
                "WANDB_MODE": run["wandb_mode"], "WANDB_RUN_ID": uuid.uuid4().hex[:8],
                "WANDB_RESUME": "never", "WANDB_RUN_GROUP": job["experiment_id"],
                "WANDB_NAME": "/".join(job[key] for key in ("experiment_id", "run_id", "job_id")),
                "WANDB_DIR": str(output / "logs" / "wandb"), "TMPDIR": "/tmp"})
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _gpu_identity(config, env, group):
    # This helper module imports only stdlib. UUIDs are resolved before torchrun starts.
    source = ("import json; from qwenvl.train.wandb_utils import prepare_training_gpu_selection; "
              "s=prepare_training_gpu_selection(); "
              "print(json.dumps({'physical_ids':s.physical_ids,'uuids':s.uuids}))")
    result = subprocess.run([config["tune"]["run"]["python"], "-c", source],
                            cwd=str(Path(config["experiment"]["repo_root"]) / "qwen-vl-finetune"),
                            env=env, capture_output=True, text=True, check=True, timeout=30)
    identity = json.loads(result.stdout)
    if identity["physical_ids"] != group or len(identity["uuids"]) != len(group):
        raise ValueError("Physical GPU identity did not match the assigned group")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(identity["uuids"])
    gpu_filter = json.dumps(group)
    env["WANDB_X_STATS_GPU_DEVICE_IDS"] = gpu_filter
    env["WANDB__STATS_GPU_DEVICE_IDS"] = gpu_filter
    return identity


def build_tune_command(job, config, inputs, output, world_size, *, master_port=None):
    """Build argv only; no shell, automatic checkpoint search or YAML reloading."""
    run, params = config["tune"]["run"], config["tune"]["params"]
    train_root = Path(config["experiment"]["repo_root"]) / "qwen-vl-finetune"
    command = [run["python"], "-m", "torch.distributed.run", "--nnodes=1",
               "--nproc_per_node={}".format(world_size), "--max_restarts=0",
               "--master_addr=" + run["master_addr"],
               "--master_port=" + str(run["master_port"] if master_port is None else master_port),
               str(train_root / "qwenvl" / "train" / "train_qwen.py")]
    arguments = {**params, "model_name_or_path": inputs["base-model"]["path"],
                 "processor_name_or_path": inputs["processor"]["path"],
                 "annotation_path": inputs["annotation-{}-train".format(job["task"])]["path"],
                 "media_root": config["experiment"]["dataset_root"],
                 "output_dir": str(output / "checkpoints"), "final_adapter_dir": str(output / "adapter"),
                 "run_name": "/".join(job[key] for key in ("experiment_id", "run_id", "job_id"))}
    for manifest in inputs.values():
        if manifest["kind"] == "checkpoint":
            arguments["resume_checkpoint_path"] = manifest["path"]
        elif manifest["kind"] == "adapter" and job["training_state_policy"] != RESTORE:
            arguments["adapter_name_or_path"] = manifest["path"]
    for key, value in arguments.items():
        if key == "deepspeed" and not value:
            continue
        command.extend(["--" + key, str(value).lower() if type(value) is bool else str(value)])
    return command


def _wait_launcher(process):
    # Leave the exited launcher unreaped until its process group is handled.
    # Its reserved PID prevents an unrelated session from reusing this PGID.
    result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def _stop_group(process):
    if process is None or process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    # Do not poll/reap the leader during the grace period: torchrun workers may
    # still be shutting down even if their launcher has already exited.
    time.sleep(5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


@contextmanager
def _signals():
    def interrupt(signum, frame):
        raise InterruptedError("Executor received signal {}".format(signum))
    previous = {number: signal.signal(number, interrupt) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _output_metadata(job, inputs, identities, deepspeed_sha256, world_size):
    return {
        "slot": job["output_spec"]["slot"], "training_task": job["task"],
        "config_fingerprint": job["config_fingerprint"], "input_fingerprints": identities,
        "base_model_fingerprint": identities["base-model"], "world_size": world_size,
        "deepspeed_sha256": deepspeed_sha256, "lineage": job["lineage"],
        "training_state_policy": job["training_state_policy"],
        "inherited_inputs": {role: value["fingerprint"] for role, value in inputs.items()
                             if value["kind"] in ("adapter", "checkpoint")},
    }


def _checkpoint_manifests(job, config, inputs, metadata, output, world_size, *, strict=True):
    """Inspect only this attempt's outputs, never search for an input checkpoint."""
    producer = {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
    params = config["tune"]["params"]
    manifests, rejected = [], []
    for checkpoint in sorted((output / "checkpoints").glob("checkpoint-*")):
        try:
            if not checkpoint.is_dir() or not checkpoint.name.removeprefix("checkpoint-").isdigit():
                raise ValueError("Unexpected checkpoint entry: {}".format(checkpoint))
            _adapter_config(checkpoint, params, inputs["base-model"]["path"])
            layout = _checkpoint_state(checkpoint, world_size, params["deepspeed"], params["fp16"])
            if checkpoint.name != "checkpoint-{}".format(layout["global_step"]):
                raise ValueError("Checkpoint directory and trainer step disagree")
            checkpoint_metadata = {**metadata, "global_step": layout["global_step"], "checkpoint_state": layout}
            manifests.append((checkpoint.name, make_manifest("checkpoint", checkpoint,
                                                            producer=producer, metadata=checkpoint_metadata)))
        except (OSError, ValueError, KeyError) as error:
            if strict:
                raise
            rejected.append({"path": str(checkpoint), "error": str(error)})
    return manifests, rejected


def _register_outputs(job_dir, job, config, inputs, identities, deepspeed_sha256, output, world_size):
    params = config["tune"]["params"]
    summary = read_json(output / "checkpoints" / "training_result.json")
    expected_policy = {FRESH: "base_model_with_fresh_training_state",
                       CONTINUE: "adapter_weights_with_fresh_training_state", RESTORE: RESTORE}
    if (type(summary.get("global_step")) is not int or summary["global_step"] <= 0
            or summary.get("train_samples", 0) <= 0 or summary.get("trainable_parameters", 0) <= 0
            or summary.get("world_size") != world_size
            or summary.get("training_state_policy") != expected_policy[job["training_state_policy"]]
            or summary.get("final_adapter_dir") != str(output / "adapter")):
        raise ValueError("Training result is incomplete or describes a different execution")
    producer = {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
    metadata = _output_metadata(job, inputs, identities, deepspeed_sha256, world_size)
    metadata["global_step"] = summary["global_step"]
    _adapter_config(output / "adapter", params, inputs["base-model"]["path"])
    checkpoints, _ = _checkpoint_manifests(job, config, inputs, metadata, output, world_size)
    if not any(role == "checkpoint-{}".format(summary["global_step"]) for role, _ in checkpoints):
        raise ValueError("Final training step has no complete resumable checkpoint")
    manifests = [("adapter", make_manifest("adapter", output / "adapter", producer=producer, metadata=metadata))]
    manifests.extend(checkpoints)
    manifests.append(("training-summary", make_manifest("summary", output / "checkpoints" / "training_result.json",
                                                       producer=producer, metadata=metadata)))
    for role, manifest in manifests:
        register_artifact(job_dir, role, manifest)
    return {role: manifest["fingerprint"] for role, manifest in manifests}


def execute_tune_job(job_dir, *, gpu_group_index=0, input_manifests=None, resume_training_state=None):
    """Run one job synchronously; parent stage/experiment scheduling is a separate layer."""
    job_dir = Path(job_dir).expanduser().resolve(strict=True)
    with _exclusive(job_dir / ".execution.lock"):
        job, snapshot, experiment_dir, run = _context(job_dir)
        config = snapshot["config"]
        groups = config["tune"]["run"]["gpu_groups"]
        if type(gpu_group_index) is not int or not 0 <= gpu_group_index < len(groups):
            raise ValueError("gpu_group_index must select one frozen tune.run.gpu_groups entry")
        group = groups[gpu_group_index]
        master_port = config["tune"]["run"]["master_port"] + gpu_group_index
        if master_port > 65535:
            raise ValueError("master_port plus GPU group index exceeds the TCP port range")
        if not group or any(type(item) is not int or item not in (1, 2, 3) for item in group):
            raise ValueError("Only explicitly assigned physical GPUs 1, 2, 3 are permitted")
        if not config["tune"]["params"]["lora_enable"] or config["tune"]["params"]["tune_mm_vision"]:
            raise ValueError("This executor requires LoRA with a frozen visual encoder")
        output = Path(job["output_dir"])
        expected_output = Path(config["experiment"]["checkpoints_root"]) / job["experiment_id"] / job["run_id"] / job["job_id"]
        if output != expected_output or output.exists() or output.is_symlink():
            raise ValueError("Refusing an unexpected or existing job output directory")
        receipt = job_dir / "execution.json"
        if receipt.exists():
            raise ValueError("An execution attempt already exists; do not overwrite its evidence")
        execution = {"schema_version": 1, "job_id": job["job_id"], "config_fingerprint": job["config_fingerprint"],
                     "status": "preparing", "prepared_at": utc_now(), "pid": os.getpid(),
                     "gpu_group_index": gpu_group_index, "physical_gpu_ids": group, "output_dir": str(output),
                     "command": None, "returncode": None, "started_at": None, "finished_at": None}
        process = None
        owns_output = False
        started = time.monotonic()
        def save_execution():
            write_json_atomic(receipt, execution)
            if owns_output:
                write_json_atomic(output / "execution.json", execution)
        save_execution()
        with _signals(), ExitStack() as stack:
            try:
                run_dir = job_dir.parent.parent
                _, parent_snapshot = load_stage_run(run_dir)
                capacity = parent_snapshot["config"]["tune"]["run"]["max_parallel_jobs"]
                execution["stage_slot"] = stack.enter_context(_stage_slot(run_dir, capacity))
                lock_root = Path("/tmp/qwen3vl-lora-tune-locks")
                for physical in sorted(group):
                    stack.enter_context(_exclusive(lock_root / "physical-{}.lock".format(physical), blocking=True))
                port_id = fingerprint([config["tune"]["run"]["master_addr"], master_port])[:24]
                stack.enter_context(_exclusive(lock_root / ("port-" + port_id + ".lock"), blocking=True))
                job, inputs, identities, ds_sha = _inputs(job, snapshot, input_manifests,
                                                         resume_training_state, experiment_dir, len(group))
                # Reject output nesting that could overwrite or rehash an input during training.
                for path in [Path(value["path"]) for value in inputs.values()] + [Path(config["experiment"]["dataset_root"]).resolve()]:
                    resolved_output = output.resolve()
                    if path == resolved_output or path in resolved_output.parents or resolved_output in path.parents:
                        raise ValueError("Output and input payloads must not overlap")
                output.mkdir(parents=True, exist_ok=False)
                owns_output = True
                (output / "logs" / "wandb").mkdir(parents=True)
                env = _environment(config, job, output, group)
                execution["gpu_identity"] = _gpu_identity(config, env, group)
                execution["command"] = build_tune_command(job, config, inputs, output, len(group), master_port=master_port)
                execution["master_port"] = master_port
                execution["deepspeed_sha256"] = ds_sha
                execution["cwd"] = str(Path(config["experiment"]["repo_root"]) / "qwen-vl-finetune")
                execution["environment"] = {key: env[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_PROJECT", "WANDB_MODE", "WANDB_RUN_ID",
                    "WANDB_RESUME", "WANDB_RUN_GROUP", "WANDB_NAME", "WANDB_DIR", "TMPDIR",
                    "WANDB_X_STATS_GPU_DEVICE_IDS", "WANDB__STATS_GPU_DEVICE_IDS")}
                execution["input_fingerprints"] = {role: value["fingerprint"] for role, value in inputs.items()}
                execution["training_state_policy"] = job["training_state_policy"]
                execution["log_path"] = str(output / "logs" / "train.log")
                _start_stage(run_dir)
                update_status(job_dir, "running", "Low-level LoRA executor started")
                execution.update(status="running", started_at=utc_now())
                save_execution()
                print("Training log: " + execution["log_path"], flush=True)
                with Path(execution["log_path"]).open("xb") as log:
                    process = subprocess.Popen(execution["command"], cwd=execution["cwd"], env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    execution["launcher_pid"] = process.pid
                    save_execution()
                    execution["returncode"] = _wait_launcher(process)
                if execution["returncode"] != 0:
                    raise RuntimeError("Training exited with code {}. See {}".format(
                        execution["returncode"], execution["log_path"]))
                process.wait()
                if config["tune"]["params"]["deepspeed"] and sha256_file(config["tune"]["params"]["deepspeed"]) != ds_sha:
                    raise ValueError("DeepSpeed configuration changed during training")
                execution["artifacts"] = _register_outputs(job_dir, job, config, inputs, identities,
                                                           ds_sha, output, len(group))
                execution.update(status="outputs_registered", finished_at=utc_now(),
                                 elapsed_seconds=time.monotonic() - started)
                save_execution()
                update_status(job_dir, "succeeded", "Training finished and adapter/checkpoint manifests registered")
                execution["status"] = "succeeded"
                save_execution()
                return execution
            except BaseException as error:
                # A second interrupt must not skip worker shutdown or record saving.
                for number in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(number, signal.SIG_IGN)
                try:
                    _stop_group(process)
                except Exception as cleanup_error:
                    execution["cleanup_error"] = str(cleanup_error)
                current, _ = load_job(job_dir)
                if process is not None and current["status"] == "running":
                    # Preserve completed checkpoint manifests from a failed/interrupted
                    # attempt so a later planned job can explicitly request full restore.
                    try:
                        if config["tune"]["params"]["deepspeed"] and sha256_file(config["tune"]["params"]["deepspeed"]) != ds_sha:
                            raise ValueError("DeepSpeed configuration changed; checkpoints require manual review")
                        metadata = _output_metadata(job, inputs, identities, ds_sha, len(group))
                        checkpoints, rejected = _checkpoint_manifests(job, config, inputs, metadata,
                                                                     output, len(group), strict=False)
                        existing = {item["role"] for item in current["artifacts"]}
                        for role, manifest in checkpoints:
                            if role not in existing:
                                register_artifact(job_dir, role, manifest)
                        execution["preserved_checkpoints"] = [role for role, _ in checkpoints]
                        execution["rejected_checkpoints"] = rejected
                    except Exception as checkpoint_error:
                        execution["checkpoint_registration_error"] = str(checkpoint_error)
                execution.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                                 finished_at=utc_now(), elapsed_seconds=time.monotonic() - started,
                                 error={"type": type(error).__name__, "message": str(error)})
                if process is not None:
                    execution["returncode"] = process.returncode
                save_execution()
                current, _ = load_job(job_dir)
                if current["status"] in ("planned", "running"):
                    update_status(job_dir, execution["status"], str(error) or type(error).__name__)
                raise
