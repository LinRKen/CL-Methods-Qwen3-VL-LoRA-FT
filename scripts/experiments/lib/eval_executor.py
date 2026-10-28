"""Execute all frozen prediction/GT pairs in one shared Qwen judge worker."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import check_manifest, make_manifest
from .common import TASKS, fingerprint, read_json, require_id, sha256_file, utc_now, write_json_atomic
from .config import resolve_job_config
from .eval_worker import inspect_evaluation, pair_id, prepare_pair
from .infer_executor import _environment, _gpu_identity
from .records import bind_inputs, load_experiment, load_job, load_stage_run, register_artifact, update_status
from .tune_executor import _exclusive, _signals, _stop_group, _wait_launcher

if __package__ == "lib":
    from pipeline import build_stage_plan
else:
    from ..pipeline import build_stage_plan


@contextmanager
def _stage_slot(run_dir, capacity):
    directory = run_dir / ".eval-slots"
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
            update_status(run_dir, "running", "Low-level batch Qwen judge started")
        except ValueError:
            current, _ = load_stage_run(run_dir)
            if current["status"] != "running":
                raise
    elif current["status"] != "running":
        raise ValueError("The parent eval run is no longer active")


def _planned_job(job, planned):
    defaults = {"task": None, "tasks": None, "depends_on": [], "requires": [],
                "input_specs": [], "lineage": {}, "output": None}
    for field, default in defaults.items():
        if job["output_spec" if field == "output" else field] != planned.get(field, default):
            raise ValueError("Job differs from its frozen planned field: " + field)


def _context(job_dir):
    job, snapshot = load_job(job_dir)
    run_dir = job_dir.parent.parent
    experiment_dir = run_dir.parents[2]
    run, run_snapshot = load_stage_run(run_dir)
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    config = snapshot["config"]
    if (job["stage"] != "eval" or run["stage"] != "eval"
            or job_dir != run_dir / "jobs" / job["job_id"]
            or run_dir != experiment_dir / "stage-runs" / "eval" / run["run_id"]
            or experiment_dir != Path(config["experiment"]["records_root"]).resolve() / job["experiment_id"]
            or experiment["experiment_id"] != job["experiment_id"]
            or run["experiment_id"] != job["experiment_id"] or run["run_id"] != job["run_id"]
            or config["experiment"] != run_snapshot["config"]["experiment"]
            or config["experiment"] != experiment_snapshot["config"]["experiment"]
            or run_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or snapshot["layers"].get("parent_snapshot") != run_snapshot["snapshot_fingerprint"]):
        raise ValueError("Eval job, run and experiment identities or snapshots disagree")
    if job["status"] != "planned" or job["artifacts"]:
        raise ValueError("Only an unexecuted planned eval job can be launched")
    if experiment["status"] != "active" or run["status"] not in ("planned", "running"):
        raise ValueError("The owning experiment and eval run must still be active")
    if {"stage": "eval", "run_id": run["run_id"],
            "path": run_dir.relative_to(experiment_dir).as_posix()} not in experiment["stage_runs"]:
        raise ValueError("Eval run is not registered with its experiment")
    if run["jobs"] != [{"job_id": job["job_id"], "path": "jobs/" + job["job_id"]}]:
        raise ValueError("Batch eval must have exactly one registered judge job")
    resolved = resolve_job_config(run_snapshot, "eval", job["job_id"],
                                  snapshot["layers"].get("job"), snapshot["layers"].get("job_cli"))
    if resolved["config_fingerprint"] != job["config_fingerprint"]:
        raise ValueError("Eval configuration cannot be reproduced from its frozen parent")
    if run["plan"] != "plan.json":
        raise ValueError("Batch eval requires a frozen stage-run plan")
    plan = read_json(run_dir / "plan.json")
    if (plan != build_stage_plan(run_snapshot, "eval") or fingerprint(plan) != run["plan_fingerprint"]
            or len(plan["jobs"]) != 1 or plan["jobs"][0]["job_id"] != job["job_id"]):
        raise ValueError("Frozen eval plan differs from its snapshot or judge job")
    _planned_job(job, plan["jobs"][0])
    if (job["task"] is not None or job["tasks"] != list(TASKS)
            or job["depends_on"] or job["training_state_policy"] is not None or job["resume_training_state"]
            or job["output_spec"] != {"artifact_kind": "metrics", "slot": "combined"}
            or job["lineage"].get("judge_model") != config["experiment"]["judge_model"]
            or job["lineage"].get("judge_model_loads") != 1):
        raise ValueError("Eval requires all scoring pairs and one judge load without training state")
    return job, snapshot, experiment_dir, run


def _prediction_source(manifest, pair, experiment_dir, experiment_config, cache):
    producer = manifest["producer"]
    if (not isinstance(producer, dict) or producer["experiment_id"] != experiment_dir.name
            or producer["stage"] != "infer" or producer["job_id"] != pair["prediction_job"]):
        raise ValueError("Prediction must come from this experiment's exact planned infer job")
    key = (producer["run_id"], producer["job_id"])
    if key not in cache:
        experiment, experiment_snapshot = load_experiment(experiment_dir)
        registrations = [item for item in experiment["stage_runs"] if item["stage"] == "infer"]
        run_dir = experiment_dir / "stage-runs" / "infer" / producer["run_id"]
        if registrations != [{"stage": "infer", "run_id": producer["run_id"],
                               "path": run_dir.relative_to(experiment_dir).as_posix()}]:
            raise ValueError("Prediction must belong to the unique registered infer run")
        run, run_snapshot = load_stage_run(run_dir)
        source_dir = run_dir / "jobs" / producer["job_id"]
        source, source_snapshot = load_job(source_dir)
        if (run["status"] != "succeeded" or source["status"] != "succeeded"
                or any(run[field] != producer[field] for field in ("experiment_id", "stage", "run_id"))
                or any(source[field] != value for field, value in producer.items())
                or run_snapshot["config"]["experiment"] != experiment_config
                or source_snapshot["config"]["experiment"] != experiment_config
                or run_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
                or source_snapshot["layers"].get("parent_snapshot") != run_snapshot["snapshot_fingerprint"]
                or {"job_id": producer["job_id"], "path": "jobs/" + producer["job_id"]} not in run["jobs"]
                or run["plan"] != "plan.json"):
            raise ValueError("Prediction producer has not succeeded under its frozen experiment")
        plan = read_json(run_dir / "plan.json")
        planned = [item for item in plan["jobs"] if item["job_id"] == source["job_id"]]
        resolved = resolve_job_config(run_snapshot, "infer", source["job_id"],
                                      source_snapshot["layers"].get("job"), source_snapshot["layers"].get("job_cli"))
        if (plan != build_stage_plan(run_snapshot, "infer") or fingerprint(plan) != run["plan_fingerprint"]
                or len(planned) != 1 or source["config_fingerprint"] != resolved["config_fingerprint"]):
            raise ValueError("Prediction producer's frozen plan or effective config differs")
        _planned_job(source, planned[0])
        # Reuse the existing success audit, including the exact execution request,
        # all source inputs, full ordered predictions, summaries and device receipt.
        # Match the advanced infer producer's canonical adapter role, as required
        # by the existing success audit; do not rewrite its frozen inputs.
        adapters = [item for item in source["inputs"].values() if item["kind"] == "adapter"]
        if len(adapters) != (0 if experiment_config["setting"] == "zero_shot" else 1):
            raise ValueError("Prediction producer has an unexpected adapter binding")
        from .infer_orchestrator import _audit_succeeded_job, _resources
        if adapters and source["inputs"].get("adapter") != adapters[0]:
            # Inspect the original roles directly instead of changing the frozen request.
            raise ValueError("Batch eval requires the infer producer's canonical adapter input role")
        _audit_succeeded_job({"directory": source_dir, "record": source, "snapshot": source_snapshot,
                              "adapter": adapters[0] if adapters else None,
                              "resources": _resources(source_snapshot["config"], source["job_id"])},
                             experiment_config["dataset_root"])
        cache[key] = (source_dir, source, source_snapshot)
    source_dir, source, source_snapshot = cache[key]
    registered = [item for item in source["artifacts"] if item["kind"] == "prediction"
                  and item["fingerprint"] == manifest["fingerprint"]
                  and read_json(source_dir / item["path"]) == manifest]
    task = pair["task"]
    expected_path = Path(source["output_dir"]) / "predictions" / (task + ".jsonl")
    metadata = manifest["metadata"]
    if (len(registered) != 1 or task not in source["tasks"]
            or source["output_spec"]["slot"] != pair["prediction_slot"]
            or manifest["requested_path"] != str(expected_path)
            or Path(manifest["path"]) != expected_path.resolve(strict=True)
            or metadata.get("slot") != pair["prediction_slot"] or metadata.get("task") != task
            or metadata.get("config_fingerprint") != source["config_fingerprint"]
            or metadata.get("lineage") != source["lineage"]
            or metadata.get("params") != source_snapshot["config"]["infer"]["params"]):
        raise ValueError("Prediction manifest is not the exact registered test artifact")
    return source, source_snapshot


def _inputs(job_dir, job, snapshot, supplied, experiment_dir):
    experiment = snapshot["config"]["experiment"]
    specs = {spec["role"]: spec for spec in job["input_specs"]}
    expected = {"judge-model": ("judge_model", experiment["judge_model"], {}),
                "judge-processor": ("processor", experiment["judge_model"], {})}
    for task in TASKS:
        expected["annotation-{}-test".format(task)] = (
            "annotation", str(Path(experiment["dataset_root"]) / task / "test.json"),
            {"task": task, "split": "test"})
    if len(specs) != len(job["input_specs"]) or set(specs) != set(expected):
        raise ValueError("Eval plan must declare judge model, processor and all test annotations")
    for role, (kind, path, metadata) in expected.items():
        source_key = "judge_model" if role in ("judge-model", "judge-processor") else "dataset_root"
        if (specs[role]["kind"] != kind or specs[role]["path"] != path
                or specs[role]["metadata"] != metadata
                or specs[role]["source"] != "configuration:experiment." + source_key):
            raise ValueError("Frozen eval input conflicts with configuration: " + role)
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
        require_id(role)
        check_manifest(manifest, verify_contents=True)
        if role not in specs and manifest["kind"] != "prediction":
            raise ValueError("Unexpected eval input role: " + role)
        if Path(manifest["requested_path"]).resolve(strict=True) != Path(manifest["path"]):
            raise ValueError("An eval input path was redirected: " + role)
    for role, spec in specs.items():
        manifest = inputs.get(role)
        if (manifest is None or manifest["kind"] != spec["kind"]
                or manifest["requested_path"] != str(Path(spec["path"]).absolute())
                or manifest["external_source"] != spec["source"] or manifest["metadata"] != spec["metadata"]):
            raise ValueError("Eval manifest disagrees with frozen specification: " + role)
    pairs = job["lineage"]["scoring_pairs"]
    predictions = {role: manifest for role, manifest in inputs.items() if manifest["kind"] == "prediction"}
    if (not pairs or len(pairs) != len(job["requires"]) or len(predictions) != len(pairs)
            or len({pair_id(pair) for pair in pairs}) != len(pairs)):
        raise ValueError("Bind exactly one distinct prediction for every planned scoring pair")
    cache, used, bound_pairs = {}, set(), []
    for pair, reference in zip(pairs, job["requires"]):
        if reference != {"stage": "infer", "job_id": pair["prediction_job"], "artifact_kind": "prediction",
                         "slot": pair["prediction_slot"], "task": pair["task"]}:
            raise ValueError("Eval requires and scoring pair order disagree")
        matches = [(role, manifest) for role, manifest in predictions.items()
                   if isinstance(manifest["producer"], dict)
                   and manifest["producer"].get("job_id") == pair["prediction_job"]
                   and manifest["metadata"].get("slot") == pair["prediction_slot"]
                   and manifest["metadata"].get("task") == pair["task"]]
        if len(matches) != 1 or matches[0][0] in used:
            raise ValueError("A scoring pair has a missing, duplicate or ambiguous prediction")
        role, prediction = matches[0]
        source, source_snapshot = _prediction_source(prediction, pair, experiment_dir, experiment, cache)
        annotation = inputs["annotation-{}-test".format(pair["task"])]
        if (source["inputs"].get("annotation-{}-test".format(pair["task"])) != annotation
                or prediction["metadata"].get("annotation_fingerprint") != annotation["fingerprint"]):
            raise ValueError("Eval GT differs from the exact annotation used by inference")
        bound = {**pair, "pair_id": pair_id(pair), "dataset": "cosine_{}_test".format(pair["task"]),
                 "prediction_role": role, "prediction_path": prediction["path"],
                 "prediction_fingerprint": prediction["fingerprint"],
                 "prediction_producer": prediction["producer"],
                 "prediction_sha256": prediction["identity"]["sha256"],
                 "annotation_path": annotation["path"], "annotation_fingerprint": annotation["fingerprint"],
                 "annotation_sha256": annotation["identity"]["sha256"],
                 "infer_params": deepcopy(source_snapshot["config"]["infer"]["params"]),
                 "output_dir": str(Path(job["output_dir"]) / "pairs" / pair_id(pair))}
        prepare_pair(bound, experiment["repo_root"])
        bound_pairs.append(bound)
        used.add(role)
    if set(predictions) != used:
        raise ValueError("Every bound prediction must belong to exactly one planned pair")
    if not job["inputs"]:
        job = bind_inputs(job_dir, inputs)
    return job, inputs, bound_pairs


def build_eval_command(config, request_path):
    worker = Path(config["experiment"]["repo_root"]) / "scripts/experiments/lib/eval_worker.py"
    return [config["eval"]["run"]["python"], str(worker), "--request", str(request_path)]


def _make_request(job, config, inputs, pairs, output, group, identity):
    source = Path(config["experiment"]["repo_root"]) / "evaluation/COSINE"
    payload = {"schema_version": 1,
               "producer": {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")},
               "config_fingerprint": job["config_fingerprint"], "lineage": job["lineage"],
               "repo_root": config["experiment"]["repo_root"], "judge_model": inputs["judge-model"]["path"],
               "judge_model_fingerprint": inputs["judge-model"]["fingerprint"],
               "judge_processor_fingerprint": inputs["judge-processor"]["fingerprint"],
               "params": deepcopy(config["eval"]["params"]), "physical_gpu_ids": group, "gpu_identity": identity,
               "input_fingerprints": {role: value["fingerprint"] for role, value in inputs.items()},
               "output_dir": str(output), "pairs": deepcopy(pairs),
               "judge_source_sha256": {name: sha256_file(source / name)
                                       for name in ("eval_cosine.py", "dataset_utils.py", "qwen_judge.py")}}
    return {**payload, "request_fingerprint": fingerprint(payload)}


def _register_outputs(job_dir, job, inputs, output, request):
    for manifest in inputs.values():
        check_manifest(manifest, verify_contents=True)
        if Path(manifest["requested_path"]).resolve(strict=True) != Path(manifest["path"]):
            raise ValueError("An eval input path changed during judging")
    summary = inspect_evaluation(request)
    common = {"config_fingerprint": job["config_fingerprint"], "lineage": job["lineage"],
              "request_fingerprint": request["request_fingerprint"],
              "input_fingerprints": request["input_fingerprints"], "params": request["params"],
              "judge_model_fingerprint": request["judge_model_fingerprint"],
              "judge_processor_fingerprint": request["judge_processor_fingerprint"],
              "prompt_version": summary["prompt_version"], "judge_model_loads": 1}
    manifests = []
    for pair, result in zip(request["pairs"], summary["pairs"]):
        metadata = {**common, **{key: pair[key] for key in ("pair_id", "prediction_job", "prediction_slot",
                    "task", "dataset", "prediction_fingerprint", "prediction_producer", "annotation_fingerprint")},
                    "sample_count": result["total"]}
        for name, kind, prefix in (("judge.raw.jsonl", "judge_raw", "judge-raw-"),
                                   ("judge.scored.jsonl", "judge_scored", "judge-scored-"),
                                   ("judge.metrics.json", "summary", "pair-metrics-")):
            role = require_id(prefix + pair["pair_id"])
            manifests.append((role, make_manifest(kind, Path(pair["output_dir"]) / name,
                             producer=request["producer"], metadata=metadata)))
    combined = {**common, "slot": "combined", "tasks": job["tasks"],
                "pair_order": summary["pair_order"], "sample_count": summary["total"],
                "combined_flat_rawacc": summary["combined_flat_rawacc"]}
    manifests.append(("metrics", make_manifest("metrics", output / "judge.metrics.json",
                     producer=request["producer"], metadata=combined)))
    manifests.append(("evaluation-summary", make_manifest("summary", output / "evaluation_result.json",
                     producer=request["producer"], metadata=combined)))
    for role, manifest in manifests:
        register_artifact(job_dir, role, manifest)
    return {role: manifest["fingerprint"] for role, manifest in manifests}


def execute_eval_job(job_dir, *, gpu_group_index=0, input_manifests=None):
    """Synchronously score all planned pairs with one shared judge-model load."""
    job_dir = Path(job_dir).expanduser().resolve(strict=True)
    with _exclusive(job_dir / ".execution.lock"):
        job, snapshot, experiment_dir, run = _context(job_dir)
        config = snapshot["config"]
        groups = config["eval"]["run"]["gpu_groups"]
        if type(gpu_group_index) is not int or not 0 <= gpu_group_index < len(groups):
            raise ValueError("gpu_group_index must select one frozen eval.run.gpu_groups entry")
        group = groups[gpu_group_index]
        if (not group or len(group) != config["eval"]["params"]["tensor_parallel_size"]
                or any(type(index) is not int or index not in (1, 2, 3) for index in group)
                or len(set(group)) != len(group)):
            raise ValueError("Assign unique physical GPUs 1, 2, 3 matching tensor_parallel_size")
        output = Path(job["output_dir"])
        expected = Path(config["experiment"]["eval_root"]) / job["experiment_id"] / job["run_id"] / job["job_id"]
        if output != expected or output.exists() or output.is_symlink():
            raise ValueError("Refusing an unexpected or existing eval output directory")
        receipt = job_dir / "execution.json"
        if receipt.exists() or receipt.is_symlink():
            raise ValueError("An execution attempt already exists; do not overwrite its evidence")
        execution = {"schema_version": 1, "job_id": job["job_id"],
                     "config_fingerprint": job["config_fingerprint"], "status": "preparing",
                     "prepared_at": utc_now(), "pid": os.getpid(), "gpu_group_index": gpu_group_index,
                     "physical_gpu_ids": group, "output_dir": str(output),
                     "tasks": job["tasks"], "scoring_pairs": job["lineage"]["scoring_pairs"],
                     "command": None, "returncode": None, "started_at": None, "finished_at": None}
        process, owns_output = None, False
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
                capacity = parent_snapshot["config"]["eval"]["run"]["max_parallel_jobs"]
                execution["stage_slot"] = stack.enter_context(_stage_slot(run_dir, capacity))
                for physical in sorted(group):
                    stack.enter_context(_exclusive(Path("/tmp/qwen3vl-lora-tune-locks") /
                                        "physical-{}.lock".format(physical), blocking=True))
                job, inputs, pairs = _inputs(job_dir, job, snapshot, input_manifests, experiment_dir)
                for path in [Path(value["path"]) for value in inputs.values()] + [
                        Path(config["experiment"]["dataset_root"]).resolve(strict=True), experiment_dir]:
                    resolved_output = output.resolve()
                    if path == resolved_output or path in resolved_output.parents or resolved_output in path.parents:
                        raise ValueError("Output must not overlap input payloads or experiment records")
                output.mkdir(parents=True, exist_ok=False)
                owns_output = True
                (output / "logs").mkdir()
                env = _environment(group)
                identity = _gpu_identity(group, env)
                request = _make_request(job, config, inputs, pairs, output, group, identity)
                request_path = job_dir / "evaluation.request.json"
                if request_path.exists() or request_path.is_symlink():
                    raise ValueError("A frozen judge request already exists")
                write_json_atomic(request_path, request)
                write_json_atomic(output / "evaluation.request.json", request)
                execution.update(gpu_identity=identity, request_fingerprint=request["request_fingerprint"],
                                 input_fingerprints=request["input_fingerprints"],
                                 judge_model_fingerprint=request["judge_model_fingerprint"],
                                 command=build_eval_command(config, request_path),
                                 cwd=config["experiment"]["repo_root"], log_path=str(output / "logs" / "eval.log"))
                execution["environment"] = {key: env[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_WORKER_MULTIPROC_METHOD",
                    "PYTHONUNBUFFERED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "TMPDIR")}
                _start_stage(run_dir)
                update_status(job_dir, "running", "Low-level batch Qwen flexible evaluation started")
                execution.update(status="running", started_at=utc_now())
                save_execution()
                print("Evaluation log: " + execution["log_path"], flush=True)
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
                    raise RuntimeError("Evaluation exited with code {}. See {}".format(
                        execution["returncode"], execution["log_path"]))
                execution["artifacts"] = _register_outputs(job_dir, job, inputs, output, request)
                execution.update(status="outputs_registered", finished_at=utc_now(),
                                 elapsed_seconds=time.monotonic() - started)
                save_execution()
                update_status(job_dir, "succeeded", "All planned judge pairs and combined metrics registered")
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
