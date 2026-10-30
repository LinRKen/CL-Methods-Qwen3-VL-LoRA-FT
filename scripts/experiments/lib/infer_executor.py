"""Execute one frozen Adapter/base inference job without importing model libraries."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import check_manifest, make_manifest
from .common import TASKS, fingerprint, read_json, utc_now, write_json_atomic
from .config import resolve_job_config
from .infer_worker import inspect_predictions, load_samples
from .records import bind_inputs, load_experiment, load_job, load_stage_run, register_artifact, update_status
from .tune_executor import (_adapter_config, _exclusive, _registered_producer,
                            _require_internal_inputs, _signals, _stop_group, _wait_launcher)
from .transfer import validate_transfer_infer_job, validate_transfer_source


@contextmanager
def _stage_slot(run_dir, capacity):
    directory = run_dir / ".infer-slots"
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
    current, _ = load_stage_run(run_dir)
    if current["status"] == "planned":
        try:
            update_status(run_dir, "running", "First low-level infer job started")
        except ValueError:
            current, _ = load_stage_run(run_dir)
            if current["status"] != "running":
                raise
    elif current["status"] != "running":
        raise ValueError("The parent infer run is no longer active")


def _context(job_dir):
    job, snapshot = load_job(job_dir)
    run_dir = job_dir.parent.parent
    experiment_dir = run_dir.parents[2]
    run, run_snapshot = load_stage_run(run_dir)
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    if (job["stage"] != "infer" or run["stage"] != "infer"
            or job_dir != run_dir / "jobs" / job["job_id"]
            or run_dir != experiment_dir / "stage-runs" / "infer" / run["run_id"]
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
    if {"stage": "infer", "run_id": run["run_id"],
            "path": run_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]:
        raise ValueError("Stage run is not registered with its experiment")
    if {"job_id": job["job_id"], "path": "jobs/" + job["job_id"]} not in run["jobs"]:
        raise ValueError("Job is not registered with its stage run")
    if run_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]:
        raise ValueError("Stage run has a different frozen parent snapshot")
    if snapshot["layers"].get("parent_snapshot") != run_snapshot["snapshot_fingerprint"]:
        raise ValueError("Job has a different frozen parent snapshot")
    resolved = resolve_job_config(run_snapshot, "infer", job["job_id"],
                                  snapshot["layers"].get("job"), snapshot["layers"].get("job_cli"))
    if resolved["config_fingerprint"] != job["config_fingerprint"]:
        raise ValueError("Job configuration cannot be reproduced from its frozen parent")
    if snapshot["config"]["experiment"] != run_snapshot["config"]["experiment"]:
        raise ValueError("Job cannot override experiment paths or model/data lineage")
    if run["plan"] != "plan.json":
        raise ValueError("The infer executor requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    if plan.get("stage") != "infer" or plan.get("config_fingerprint") != run["config_fingerprint"]:
        raise ValueError("Frozen plan does not describe this infer run")
    planned = [item for item in plan["jobs"] if item["job_id"] == job["job_id"]]
    if len(planned) != 1:
        raise ValueError("Job must occur exactly once in the frozen plan")
    defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                "input_specs": [], "lineage": {}, "output": None}
    for field, default in defaults.items():
        actual = job["output_spec" if field == "output" else field]
        if actual != planned[0].get(field, default):
            raise ValueError("Job no longer matches planned field: " + field)
    if (job["task"] is not None or not isinstance(job["tasks"], list) or not job["tasks"]
            or any(task not in TASKS for task in job["tasks"])
            or len(set(job["tasks"])) != len(job["tasks"])
            or not job["output_spec"] or job["output_spec"].get("artifact_kind") != "prediction"
            or job["output_spec"].get("tasks") != job["tasks"]
            or job["lineage"].get("test_tasks") != job["tasks"]
            or job["lineage"].get("model_loads_per_job") != 1
            or job["training_state_policy"] is not None or job["resume_training_state"]):
        raise ValueError("An infer job requires distinct test tasks and one model load, without training state")
    for dependency in job["depends_on"]:
        sibling, _ = load_job(run_dir / "jobs" / dependency)
        if (sibling["status"] != "succeeded" or sibling["run_id"] != job["run_id"]
                or sibling["experiment_id"] != job["experiment_id"]):
            raise ValueError("An inference dependency has not succeeded: " + dependency)
    return job, snapshot, experiment_dir, run




def _inputs(job_dir, job, snapshot, supplied, experiment_dir):
    experiment = snapshot["config"]["experiment"]
    specs = {spec["role"]: spec for spec in job["input_specs"]}
    expected = {"base-model": ("base_model", experiment["base_model"], {}),
                "processor": ("processor", experiment["processor_path"], {})}
    for task in job["tasks"]:
        expected["annotation-{}-test".format(task)] = (
            "annotation", str(Path(experiment["dataset_root"]) / task / "test.json"),
            {"task": task, "split": "test"})
    if len(specs) != len(job["input_specs"]) or set(specs) != set(expected):
        raise ValueError("Infer plan must declare the base, processor and every test annotation")
    for role, (kind, path, metadata) in expected.items():
        source_key = {"base-model": "base_model", "processor": "processor_path"}.get(role, "dataset_root")
        if (specs[role]["kind"] != kind or specs[role]["path"] != path
                or specs[role]["metadata"] != metadata
                or specs[role]["source"] != "configuration:experiment." + source_key):
            raise ValueError("Frozen input specification conflicts with configuration: " + role)
    if job["inputs"]:
        if supplied:
            raise ValueError("Bound inputs cannot be overridden")
        inputs = deepcopy(job["inputs"])
    else:
        inputs = deepcopy(supplied or {})
        for role, spec in specs.items():
            if role not in inputs:
                inputs[role] = make_manifest(spec["kind"], spec["path"],
                                            source=spec["source"], metadata=spec["metadata"])
    for role, manifest in inputs.items():
        check_manifest(manifest, verify_contents=True)
        if role not in specs and manifest["kind"] != "adapter":
            raise ValueError("Unexpected infer input role: " + role)
    for role, spec in specs.items():
        manifest = inputs.get(role)
        if (manifest is None or manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != str(Path(spec["path"]).absolute())
                or manifest["external_source"] != spec["source"]
                or manifest["metadata"] != spec["metadata"]
                or Path(spec["path"]).resolve(strict=True) != Path(manifest["path"])):
            raise ValueError("Input manifest disagrees with frozen specification: " + role)
    adapters = [manifest for manifest in inputs.values() if manifest["kind"] == "adapter"]
    expected_adapters = 0 if experiment["setting"] == "zero_shot" else 1
    if len(adapters) != expected_adapters or len(job["requires"]) != expected_adapters:
        raise ValueError("Infer requires exactly one planned adapter, or none for Zero-shot")
    if experiment["setting"] == "transfer_round2" and experiment.get("transfer_source_experiment_id"):
        source_context = validate_transfer_source(
            snapshot, job["experiment_id"], audit_adapters=False, require_frozen=True,
            adapter_job_id=job["lineage"].get("adapter_ref"))
        validate_transfer_infer_job(snapshot, job, source_context, adapters[0])
    if adapters:
        reference = job["requires"][0]
        expected_source = job["lineage"].get("adapter_source_experiment_id", job["experiment_id"])
        reference_source = reference.get("experiment_id", job["experiment_id"])
        if reference_source != expected_source:
            raise ValueError("Infer adapter source differs from its frozen lineage")
    _require_internal_inputs(job, inputs, experiment_dir)
    for adapter in adapters:
        source = _registered_producer(adapter, experiment["records_root"])
        producer = adapter["producer"]
        source_dir = (Path(experiment["records_root"]) / producer["experiment_id"] / "stage-runs" / "tune"
                      / producer["run_id"] / "jobs" / producer["job_id"])
        source_job, source_snapshot = load_job(source_dir)
        _adapter_config(adapter["path"], source_snapshot["config"]["tune"]["params"],
                        inputs["base-model"]["path"])
        metadata = adapter["metadata"]
        if (job["lineage"].get("adapter_ref") != source["job_id"]
                or metadata.get("base_model_fingerprint") != inputs["base-model"]["fingerprint"]
                or metadata.get("config_fingerprint") != source["config_fingerprint"]
                or metadata.get("lineage") != source["lineage"]
                or metadata.get("training_state_policy") != source["training_state_policy"]
                or source_job["inputs"].get("base-model", {}).get("fingerprint") != inputs["base-model"]["fingerprint"]):
            raise ValueError("Adapter source, base identity or training lineage does not match")
    if not adapters and (job["lineage"].get("model") != experiment["base_model"]
                         or job["lineage"].get("adapter") is not None):
        raise ValueError("Zero-shot lineage must identify the frozen base without an adapter")
    if not job["inputs"]:
        job = bind_inputs(job_dir, inputs)
    return job, inputs


def _environment(group):
    env = os.environ.copy()
    for key in list(env):
        if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
            del env[key]
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(map(str, group)), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "TMPDIR": "/tmp"})
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _gpu_identity(group, env):
    # NVML indices must agree with CUDA's PCI order before numeric visibility is used.
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
                             "--format=csv,noheader,nounits"],
                            env=env, capture_output=True, text=True, check=True, timeout=15)
    rows = [tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    if (not rows or any(len(row) != 3 for row in rows)
            or len({row[0] for row in rows}) != len(rows)
            or len({row[1] for row in rows}) != len(rows)):
        raise ValueError("Cannot verify physical GPU identity")
    pci_order = sorted(rows, key=lambda row: tuple(
        int(part, 16) for part in row[2].replace(".", ":").split(":")))
    if [row[0] for row in pci_order] != [str(index) for index in range(len(rows))]:
        raise ValueError("Physical GPU indices differ from CUDA PCI ordering")
    identities = {int(index): {"uuid": uuid, "pci_bus_id": pci} for index, uuid, pci in rows}
    if any(index not in identities for index in group):
        raise ValueError("An explicitly selected GPU is absent")
    return {"physical_ids": group, "uuids": [identities[index]["uuid"] for index in group],
            "pci_bus_ids": [identities[index]["pci_bus_id"] for index in group]}


def build_infer_command(config, request_path):
    """Build a single argv; the worker consumes a frozen, fingerprinted request."""
    worker = Path(config["experiment"]["repo_root"]) / "scripts/experiments/lib/infer_worker.py"
    return [config["infer"]["run"]["python"], str(worker), "--request", str(request_path)]


def _make_request(job, config, inputs, output, group, identity):
    adapters = [item for item in inputs.values() if item["kind"] == "adapter"]
    payload = {"schema_version": 1,
               "producer": {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")},
               "config_fingerprint": job["config_fingerprint"], "lineage": job["lineage"],
               "repo_root": config["experiment"]["repo_root"],
               "media_root": str(Path(config["experiment"]["dataset_root"]).resolve(strict=True)),
               "base_model": inputs["base-model"]["path"],
               "processor_path": inputs["processor"]["path"],
               "adapter_path": adapters[0]["path"] if adapters else None,
               "adapter_fingerprint": adapters[0]["fingerprint"] if adapters else None,
               "params": deepcopy(config["infer"]["params"]),
               "physical_gpu_ids": group, "gpu_identity": identity,
               "input_fingerprints": {role: value["fingerprint"] for role, value in inputs.items()},
               "output_dir": str(output), "tasks": []}
    for task in job["tasks"]:
        annotation = inputs["annotation-{}-test".format(task)]
        payload["tasks"].append({"task": task, "dataset": "cosine_{}_test".format(task),
                                 "annotation_path": annotation["path"],
                                 "annotation_sha256": annotation["identity"]["sha256"],
                                 "output_path": str(output / "predictions" / (task + ".jsonl"))})
    return {**payload, "request_fingerprint": fingerprint(payload)}


def _register_outputs(job_dir, job, config, inputs, output, request):
    summary = read_json(output / "inference_result.json")
    tasks = [item["task"] for item in request["tasks"]]
    if (summary.get("schema_version") != 1 or summary.get("status") != "succeeded"
            or summary.get("request_fingerprint") != request["request_fingerprint"]
            or summary.get("producer") != request["producer"]
            or summary.get("config_fingerprint") != job["config_fingerprint"]
            or summary.get("input_fingerprints") != request["input_fingerprints"]
            or summary.get("params") != request["params"]
            or summary.get("physical_gpu_ids") != request["physical_gpu_ids"]
            or summary.get("model_loads") != 1 or summary.get("engine_shutdown") is not True
            or summary.get("task_order") != tasks or not isinstance(summary.get("tasks"), list)
            or len(summary["tasks"]) != len(tasks)):
        raise ValueError("Inference summary is incomplete or describes another execution")
    for manifest in inputs.values():
        check_manifest(manifest, verify_contents=True)
        if Path(manifest["requested_path"]).resolve(strict=True) != Path(manifest["path"]):
            raise ValueError("An input path was redirected during inference")
    manifests = []
    for task_spec, task_summary in zip(request["tasks"], summary["tasks"]):
        task = task_spec["task"]
        samples = load_samples(task_spec["annotation_path"], request["media_root"])
        inspected = inspect_predictions(task_spec, samples, request["params"])
        if any(task_summary.get(key) != value for key, value in inspected.items()):
            raise ValueError("Prediction summary differs from its complete test payload: " + task)
        metadata = {"slot": job["output_spec"]["slot"], "task": task, "split": "test",
                    "dataset": task_spec["dataset"], "sample_count": inspected["sample_count"],
                    "empty_predictions": inspected["empty_predictions"],
                    "config_fingerprint": job["config_fingerprint"], "lineage": job["lineage"],
                    "input_fingerprints": request["input_fingerprints"],
                    "base_model_fingerprint": inputs["base-model"]["fingerprint"],
                    "processor_fingerprint": inputs["processor"]["fingerprint"],
                    "adapter_fingerprint": request["adapter_fingerprint"],
                    "annotation_fingerprint": inputs["annotation-{}-test".format(task)]["fingerprint"],
                    "request_fingerprint": request["request_fingerprint"],
                    "params": request["params"], "model_loads_per_job": 1}
        manifests.append(("prediction-" + task, make_manifest("prediction", task_spec["output_path"],
                         producer=request["producer"], metadata=metadata)))
    manifests.append(("inference-summary", make_manifest("summary", output / "inference_result.json",
                     producer=request["producer"], metadata={"slot": job["output_spec"]["slot"],
                     "tasks": tasks, "config_fingerprint": job["config_fingerprint"],
                     "request_fingerprint": request["request_fingerprint"]})))
    for role, manifest in manifests:
        register_artifact(job_dir, role, manifest)
    return {role: manifest["fingerprint"] for role, manifest in manifests}


def execute_infer_job(job_dir, *, gpu_group_index=0, input_manifests=None):
    """Synchronously execute one Adapter/base and all its frozen test tasks."""
    job_dir = Path(job_dir).expanduser().resolve(strict=True)
    with _exclusive(job_dir / ".execution.lock"):
        job, snapshot, experiment_dir, run = _context(job_dir)
        config = snapshot["config"]
        groups = config["infer"]["run"]["gpu_groups"]
        if type(gpu_group_index) is not int or not 0 <= gpu_group_index < len(groups):
            raise ValueError("gpu_group_index must select one frozen infer.run.gpu_groups entry")
        group = groups[gpu_group_index]
        if (not group or len(group) != config["infer"]["params"]["tensor_parallel_size"]
                or any(type(index) is not int or index not in (1, 2, 3) for index in group)
                or len(set(group)) != len(group)):
            raise ValueError("Assign unique physical GPUs 1, 2, 3 matching tensor_parallel_size")
        output = Path(job["output_dir"])
        expected = Path(config["experiment"]["infer_root"]) / job["experiment_id"] / job["run_id"] / job["job_id"]
        if output != expected or output.exists() or output.is_symlink():
            raise ValueError("Refusing an unexpected or existing infer output directory")
        receipt = job_dir / "execution.json"
        if receipt.exists() or receipt.is_symlink():
            raise ValueError("An execution attempt already exists; do not overwrite its evidence")
        execution = {"schema_version": 1, "job_id": job["job_id"],
                     "config_fingerprint": job["config_fingerprint"], "status": "preparing",
                     "prepared_at": utc_now(), "pid": os.getpid(), "gpu_group_index": gpu_group_index,
                     "physical_gpu_ids": group, "output_dir": str(output), "tasks": job["tasks"],
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
                capacity = parent_snapshot["config"]["infer"]["run"]["max_parallel_jobs"]
                execution["stage_slot"] = stack.enter_context(_stage_slot(run_dir, capacity))
                for physical in sorted(group):
                    stack.enter_context(_exclusive(Path("/tmp/qwen3vl-lora-tune-locks") /
                                        "physical-{}.lock".format(physical), blocking=True))
                job, inputs = _inputs(job_dir, job, snapshot, input_manifests, experiment_dir)
                for path in [Path(value["path"]) for value in inputs.values()] + [
                        Path(config["experiment"]["dataset_root"]).resolve(strict=True), job_dir]:
                    resolved_output = output.resolve()
                    if path == resolved_output or path in resolved_output.parents or resolved_output in path.parents:
                        raise ValueError("Output must not overlap input payloads or job records")
                output.mkdir(parents=True, exist_ok=False)
                owns_output = True
                (output / "logs").mkdir()
                env = _environment(group)
                identity = _gpu_identity(group, env)
                request = _make_request(job, config, inputs, output, group, identity)
                request_path = job_dir / "inference.request.json"
                if request_path.exists() or request_path.is_symlink():
                    raise ValueError("A frozen worker request already exists")
                write_json_atomic(request_path, request)
                write_json_atomic(output / "inference.request.json", request)
                execution.update(gpu_identity=identity, request_fingerprint=request["request_fingerprint"],
                                 input_fingerprints=request["input_fingerprints"],
                                 adapter_fingerprint=request["adapter_fingerprint"],
                                 command=build_infer_command(config, request_path),
                                 cwd=str(Path(config["experiment"]["repo_root"]) / "qwen-vl-finetune"),
                                 log_path=str(output / "logs" / "infer.log"))
                execution["environment"] = {key: env[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_WORKER_MULTIPROC_METHOD",
                    "PYTHONUNBUFFERED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "TMPDIR")}
                _start_stage(run_dir)
                update_status(job_dir, "running", "Low-level Adapter/base vLLM executor started")
                execution.update(status="running", started_at=utc_now())
                save_execution()
                print("Inference log: " + execution["log_path"], flush=True)
                with Path(execution["log_path"]).open("xb") as log:
                    process = subprocess.Popen(execution["command"], cwd=execution["cwd"], env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    execution["launcher_pid"] = process.pid
                    save_execution()
                    execution["returncode"] = _wait_launcher(process)
                _stop_group(process)
                execution["cleanup_completed"] = True
                if execution["returncode"] != 0:
                    raise RuntimeError("Inference exited with code {}. See {}".format(
                        execution["returncode"], execution["log_path"]))
                execution["artifacts"] = _register_outputs(job_dir, job, config, inputs, output, request)
                execution.update(status="outputs_registered", finished_at=utc_now(),
                                 elapsed_seconds=time.monotonic() - started)
                save_execution()
                update_status(job_dir, "succeeded", "Every planned test prediction and summary registered")
                execution["status"] = "succeeded"
                save_execution()
                return execution
            except BaseException as error:
                for number in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(number, signal.SIG_IGN)
                try:
                    _stop_group(process)
                    execution["cleanup_completed"] = True
                except Exception as cleanup_error:
                    execution["cleanup_error"] = str(cleanup_error)
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
