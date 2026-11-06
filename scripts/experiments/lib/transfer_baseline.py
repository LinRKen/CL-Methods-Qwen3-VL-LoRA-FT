"""Resolve the six successful round-one adapters for a Transfer baseline."""

from pathlib import Path

from .artifacts import ManifestVerificationCache, check_manifest, make_manifest
from .common import TASKS, fingerprint, read_json
from .config import resolve_job_config
from .records import load_experiment, load_job, load_stage_run
from .tune_orchestrator import _successful_adapter


def _pipeline_helpers():
    if __package__ == "lib":
        from pipeline import build_stage_plan
    else:
        from ..pipeline import build_stage_plan
    return build_stage_plan


def validate_transfer_baseline_source(snapshot, experiment_id=None):
    """Verify the declared Transfer-round1 tune run and bind its six first-round adapters."""
    config = snapshot["config"]
    experiment = config["experiment"]
    source_id = experiment.get("transfer_source_experiment_id")
    if source_id is None or source_id == experiment_id:
        raise ValueError("Baseline requires a distinct transfer_round1 source experiment")
    source_dir = Path(experiment["records_root"]).resolve() / source_id
    source_record, source_snapshot = load_experiment(source_dir)
    source_experiment = source_snapshot["config"]["experiment"]
    if (source_record["experiment_id"] != source_id
            or source_record["status"] not in ("active", "succeeded")
            or source_experiment["setting"] != "transfer_round1"):
        raise ValueError("Baseline source must be the declared transfer_round1 experiment")
    for key in ("repo_root", "base_model", "processor_path", "dataset_root", "tasks", "task_order"):
        if source_experiment[key] != experiment[key]:
            raise ValueError("Baseline and source experiment differ in experiment.{}".format(key))
    registrations = [item for item in source_record["stage_runs"] if item["stage"] == "tune"]
    if len(registrations) != 1:
        raise ValueError("Baseline source must have one registered tune run")
    registration = registrations[0]
    tune_dir = source_dir / "stage-runs/tune" / registration["run_id"]
    if registration.get("path") != tune_dir.relative_to(source_dir).as_posix():
        raise ValueError("Baseline source tune registration path is inconsistent")
    tune_run, tune_snapshot = load_stage_run(tune_dir)
    if (tune_run["stage"] != "tune" or tune_run["status"] != "succeeded"
            or tune_run["experiment_id"] != source_id or tune_run["run_id"] != registration["run_id"]
            or tune_snapshot["layers"].get("parent_snapshot") != source_snapshot["snapshot_fingerprint"]):
        raise ValueError("Baseline source tune run has not succeeded as a frozen child")
    build_stage_plan = _pipeline_helpers()
    plan = read_json(tune_dir / "plan.json")
    if (plan != build_stage_plan(tune_snapshot, "tune")
            or tune_run["plan_fingerprint"] != fingerprint(plan)
            or len(plan["jobs"]) != 36 or len(tune_run["jobs"]) != 36):
        raise ValueError("Baseline source must have the complete frozen 36-job Transfer tune plan")
    planned = {item["job_id"]: item for item in plan["jobs"]}
    registered = {item["job_id"]: item for item in tune_run["jobs"]}
    expected_tasks = {task: "tune-round1-{}".format(task) for task in TASKS}
    cache, adapters = ManifestVerificationCache(), {}
    for task, job_id in expected_tasks.items():
        if (job_id not in planned or registered.get(job_id, {}).get("path") != "jobs/" + job_id):
            raise ValueError("Baseline first-round tune job is not registered: " + job_id)
        job_dir = tune_dir / "jobs" / job_id
        job, job_snapshot = load_job(job_dir)
        resolved = resolve_job_config(tune_snapshot, "tune", job_id,
                                      job_snapshot["layers"].get("job"),
                                      job_snapshot["layers"].get("job_cli"))
        item = planned[job_id]
        if (job["status"] != "succeeded" or job["task"] != task
                or job["config_fingerprint"] != resolved["config_fingerprint"]
                or job_snapshot["layers"].get("parent_snapshot") != tune_snapshot["snapshot_fingerprint"]
                or job["lineage"] != item["lineage"] or job["output_spec"] != item["output"]
                or job["requires"] != item["requires"] or job["input_specs"] != item["input_specs"]):
            raise ValueError("Baseline first-round job differs from its frozen plan: " + job_id)
        adapter = _successful_adapter(job_dir, job, verify_contents=True)
        manifest = check_manifest(read_json(adapter["manifest"]), verify_contents=True,
                                  verification_cache=cache)
        adapters[task] = {"job_id": job_id, "job_config_fingerprint": job["config_fingerprint"],
                          "manifest_path": adapter["manifest"], "manifest": manifest}
    return {"experiment_id": source_id,
            "experiment_snapshot_fingerprint": source_snapshot["snapshot_fingerprint"],
            "tune_run_id": tune_run["run_id"],
            "tune_snapshot_fingerprint": tune_snapshot["snapshot_fingerprint"],
            "tune_plan_fingerprint": tune_run["plan_fingerprint"],
            "adapters": adapters}


def bind_transfer_baseline_snapshot(snapshot, experiment_id=None):
    from .config import make_snapshot

    source = validate_transfer_baseline_source(snapshot, experiment_id)
    predictions = discover_reusable_predictions(snapshot, source)
    evaluations = discover_reusable_evaluations(snapshot, predictions)
    return make_snapshot(snapshot["config"], snapshot["sources"],
                         {**snapshot["layers"], "transfer_baseline_source": source,
                          "transfer_baseline_reuse": {"predictions": predictions,
                                                      "evaluations": evaluations}})


def _experiment_ids(snapshot):
    root = Path(snapshot["config"]["experiment"]["records_root"]).resolve()
    return sorted(path.name for path in root.iterdir()
                  if path.is_dir() and (path / "experiment.json").is_file())


def _infer_source_info(snapshot, candidate, source, task, *, verification_cache=None):
    """Re-open and audit one frozen prediction producer before importing its artifact."""
    from .infer_orchestrator import _artifact_map, _audit_succeeded_job, _resources

    experiment_config = snapshot["config"]["experiment"]
    producer = candidate["manifest"]["producer"]
    if (candidate.get("source", {}).get("experiment_id") != producer.get("experiment_id")
            or candidate.get("source", {}).get("job_id") != producer.get("job_id")
            or producer.get("stage") != "infer"):
        raise ValueError("Reusable prediction producer differs from its frozen source")
    experiment_dir = Path(experiment_config["records_root"]).resolve() / producer["experiment_id"]
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    registrations = [item for item in experiment["stage_runs"] if item["stage"] == "infer"]
    if (experiment["status"] not in ("active", "succeeded") or len(registrations) != 1
            or experiment_snapshot["config"]["experiment"]["setting"] == "transfer_baseline"):
        raise ValueError("Reusable prediction must belong to one registered non-baseline infer run")
    registration = registrations[0]
    run_dir = experiment_dir / "stage-runs/infer" / registration["run_id"]
    run, run_snapshot = load_stage_run(run_dir)
    job_dir = run_dir / "jobs" / producer["job_id"]
    job, job_snapshot = load_job(job_dir)
    plan = read_json(run_dir / "plan.json")
    from .pipeline import build_stage_plan
    if (registration.get("path") != run_dir.relative_to(experiment_dir).as_posix()
            or producer != {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
            or run["status"] != "succeeded" or job["status"] != "succeeded"
            or run_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or job_snapshot["layers"].get("parent_snapshot") != run_snapshot["snapshot_fingerprint"]
            or plan != build_stage_plan(run_snapshot, "infer")
            or run["plan_fingerprint"] != fingerprint(plan)
            or job_snapshot["config"]["infer"]["params"] != snapshot["config"]["infer"]["params"]
            or task not in job["tasks"]):
        raise ValueError("Reusable prediction producer differs from its succeeded frozen plan")
    planned = [item for item in plan["jobs"] if item["job_id"] == job["job_id"]]
    resolved = resolve_job_config(run_snapshot, "infer", job["job_id"],
                                  job_snapshot["layers"].get("job"), job_snapshot["layers"].get("job_cli"))
    if (len(planned) != 1 or job["config_fingerprint"] != resolved["config_fingerprint"]
            or {"job_id": job["job_id"], "path": "jobs/" + job["job_id"]} not in run["jobs"]):
        raise ValueError("Reusable prediction job config or registration is inconsistent")
    inputs = job["inputs"]
    source_task = candidate.get("source_task")
    if source_task not in source["adapters"]:
        raise ValueError("Reusable prediction lacks its first-round source task")
    source_adapter = source["adapters"][source_task]["manifest"]
    expected_base = make_manifest("base_model", experiment_config["base_model"],
                                  source="configuration:experiment.base_model",
                                  verification_cache=verification_cache)
    expected_processor = make_manifest("processor", experiment_config["processor_path"],
                                       source="configuration:experiment.processor_path",
                                       verification_cache=verification_cache)
    expected_annotation = make_manifest(
        "annotation", Path(experiment_config["dataset_root"]) / task / "test.json",
        source="configuration:experiment.dataset_root", metadata={"task": task, "split": "test"},
        verification_cache=verification_cache)
    if (inputs.get("base-model", {}).get("fingerprint") != expected_base["fingerprint"]
            or inputs.get("processor", {}).get("fingerprint") != expected_processor["fingerprint"]
            or inputs.get("adapter", {}).get("identity") != source_adapter["identity"]
            or candidate.get("adapter_identity") != source_adapter["identity"]
            or inputs.get("annotation-{}-test".format(task)) != expected_annotation
            or candidate.get("annotation_fingerprint") != expected_annotation["fingerprint"]
            or candidate.get("infer_params") != snapshot["config"]["infer"]["params"]):
        raise ValueError("Reusable prediction model, adapter, annotation or parameters differ")
    adapter_manifest = inputs["adapter"]
    adapter_info = {"manifest": str(job_dir / "artifacts/adapter.json"),
                    "path": adapter_manifest["path"], "fingerprint": adapter_manifest["fingerprint"],
                    "producer": adapter_manifest["producer"], "slot": adapter_manifest["metadata"]["slot"]}
    _audit_succeeded_job({"directory": job_dir, "record": job, "snapshot": job_snapshot,
                          "adapter": adapter_info, "resources": _resources(job_snapshot["config"], job["job_id"])},
                         experiment_config["dataset_root"], verification_cache=verification_cache)
    outputs = _artifact_map(job_dir, job, verification_cache=verification_cache)
    actual = outputs.get("prediction-" + task)
    if actual != candidate["manifest"]:
        raise ValueError("Frozen reusable prediction is not the registered producer artifact")
    return {"directory": job_dir, "record": job, "snapshot": job_snapshot,
            "manifest": actual, "manifest_path": str(job_dir / "artifacts" / ("prediction-" + task + ".json"))}


def discover_reusable_predictions(snapshot, source):
    """Find prior predictions whose exact first-round adapter/config/GT match."""
    from .infer_orchestrator import _artifact_map, _audit_succeeded_job, _resources

    config, experiment = snapshot["config"], snapshot["config"]["experiment"]
    cache, selected = ManifestVerificationCache(), {}
    adapters = {task: item["manifest"] for task, item in source["adapters"].items()}
    base = make_manifest("base_model", experiment["base_model"],
                         source="configuration:experiment.base_model", verification_cache=cache)
    processor = make_manifest("processor", experiment["processor_path"],
                              source="configuration:experiment.processor_path", verification_cache=cache)
    annotation_manifests = {
        task: make_manifest("annotation", Path(experiment["dataset_root"]) / task / "test.json",
                            source="configuration:experiment.dataset_root",
                            metadata={"task": task, "split": "test"}, verification_cache=cache)
        for task in TASKS
    }
    source_experiment_id = experiment["transfer_source_experiment_id"]
    for experiment_id in _experiment_ids(snapshot):
        if experiment_id == source_experiment_id:
            continue
        try:
            directory = Path(experiment["records_root"]).resolve() / experiment_id
            record, exp_snapshot = load_experiment(directory)
            exp_config = exp_snapshot["config"]["experiment"]
            if (record["status"] not in ("active", "succeeded")
                    or exp_config["setting"] == "transfer_baseline"
                    or any(exp_config[key] != experiment[key] for key in
                           ("repo_root", "base_model", "processor_path", "dataset_root", "tasks", "task_order"))):
                continue
            registrations = [item for item in record["stage_runs"] if item["stage"] == "infer"]
            if len(registrations) != 1:
                continue
            registration = registrations[0]
            run_dir = directory / "stage-runs/infer" / registration["run_id"]
            run, run_snapshot = load_stage_run(run_dir)
            if (registration.get("path") != run_dir.relative_to(directory).as_posix()
                    or run["status"] != "succeeded"
                    or run_snapshot["layers"].get("parent_snapshot") != exp_snapshot["snapshot_fingerprint"]):
                continue
            from .pipeline import build_stage_plan
            plan = read_json(run_dir / "plan.json")
            if (plan != build_stage_plan(run_snapshot, "infer")
                    or run["plan_fingerprint"] != fingerprint(plan)
                    or run["jobs"] != [{"job_id": item["job_id"], "path": "jobs/" + item["job_id"]}
                                        for item in plan["jobs"]]):
                continue
            for planned in plan["jobs"]:
                job_dir = run_dir / "jobs" / planned["job_id"]
                job, job_snapshot = load_job(job_dir)
                if (job["status"] != "succeeded"
                        or job_snapshot["config"]["infer"]["params"] != config["infer"]["params"]):
                    continue
                adapter = job["inputs"].get("adapter")
                if adapter is None:
                    continue
                source_tasks = [task for task, manifest in adapters.items()
                                if manifest["identity"] == adapter.get("identity")
                                and manifest["metadata"].get("base_model_fingerprint")
                                    == adapter.get("metadata", {}).get("base_model_fingerprint")]
                if len(source_tasks) != 1:
                    continue
                source_task = source_tasks[0]
                if (job["inputs"].get("base-model") != base
                        or job["inputs"].get("processor") != processor):
                    continue
                adapter_config = job_snapshot["config"]["tune"]["params"]
                from .infer_executor import _adapter_config
                _adapter_config(adapter["path"], adapter_config, base["path"])
                adapter_info = {"manifest": str(job_dir / "artifacts/adapter.json"),
                                "path": adapter["path"], "fingerprint": adapter["fingerprint"],
                                "producer": adapter["producer"], "slot": adapter["metadata"]["slot"]}
                info = {"directory": job_dir, "record": job, "snapshot": job_snapshot,
                        "adapter": adapter_info,
                        "resources": _resources(job_snapshot["config"], job["job_id"])}
                _audit_succeeded_job(info, experiment["dataset_root"], verification_cache=cache)
                artifacts = _artifact_map(job_dir, job, verification_cache=cache)
                for task in job["tasks"]:
                    if task not in TASKS or job["inputs"].get("annotation-{}-test".format(task)) != annotation_manifests[task]:
                        continue
                    manifest = artifacts.get("prediction-" + task)
                    if manifest is None:
                        continue
                    key = source_task + "--" + task
                    selected.setdefault(key, {"manifest": manifest,
                        "manifest_path": str(job_dir / "artifacts" / ("prediction-" + task + ".json")),
                        "source": {"experiment_id": experiment_id, "run_id": run["run_id"],
                                   "job_id": job["job_id"], "setting": exp_config["setting"],
                                   "source_task": source_task},
                        "annotation_fingerprint": annotation_manifests[task]["fingerprint"],
                        "infer_params": job_snapshot["config"]["infer"]["params"],
                        "base_model_fingerprint": base["fingerprint"],
                        "processor_fingerprint": processor["fingerprint"],
                        "adapter_identity": adapter["identity"]})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return selected


def discover_reusable_evaluations(snapshot, reusable_predictions):
    """Find already-audited pair scores for the exact reusable prediction and judge."""
    from .common import sha256_file
    from .eval_orchestrator import _audit_succeeded_job, _bind_predictions, _resources
    from .eval_worker import pair_id
    from .pipeline import build_stage_plan

    experiment = snapshot["config"]["experiment"]
    cache, selected = ManifestVerificationCache(), {}
    judge = make_manifest("judge_model", experiment["judge_model"],
                          source="configuration:experiment.judge_model", verification_cache=cache)
    judge_processor = make_manifest("processor", experiment["judge_model"],
                                    source="configuration:experiment.judge_model", verification_cache=cache)
    code_dir = Path(experiment["repo_root"]) / "evaluation/COSINE"
    code_hashes = {name: sha256_file(code_dir / name)
                   for name in ("eval_cosine.py", "dataset_utils.py", "qwen_judge.py")}
    for experiment_id in _experiment_ids(snapshot):
        try:
            directory = Path(experiment["records_root"]).resolve() / experiment_id
            record, exp_snapshot = load_experiment(directory)
            if (record["status"] not in ("active", "succeeded")
                    or exp_snapshot["config"]["experiment"]["setting"] == "transfer_baseline"):
                continue
            registrations = [item for item in record["stage_runs"] if item["stage"] == "eval"]
            if len(registrations) != 1:
                continue
            registration = registrations[0]
            run_dir = directory / "stage-runs/eval" / registration["run_id"]
            run, run_snapshot = load_stage_run(run_dir)
            if registration.get("path") != run_dir.relative_to(directory).as_posix() or run["status"] != "succeeded":
                continue
            plan = read_json(run_dir / "plan.json")
            if plan != build_stage_plan(run_snapshot, "eval") or run["plan_fingerprint"] != fingerprint(plan):
                continue
            job_dir = run_dir / "jobs/eval-judge-all"
            job, job_snapshot = load_job(job_dir)
            if (job["status"] != "succeeded"
                    or job_snapshot["config"]["eval"]["params"] != snapshot["config"]["eval"]["params"]):
                continue
            info = {"directory": job_dir, "record": job, "snapshot": job_snapshot,
                    "initial_status": "succeeded", "resources": _resources(job_snapshot["config"])}
            _bind_predictions(info, directory, record, exp_snapshot)
            _audit_succeeded_job(info, directory, verification_cache=cache)
            request = info["request"]
            if (request["judge_model_fingerprint"] != judge["fingerprint"]
                    or request["judge_processor_fingerprint"] != judge_processor["fingerprint"]
                    or request.get("judge_source_sha256") != code_hashes):
                continue
            request_path = job_dir / "evaluation.request.json"
            request_sha = sha256_file(request_path)
            request_pairs = {pair_id(pair): pair for pair in request["pairs"]}
            for pair_result in info["result"]["pairs"]:
                source_pair = request_pairs.get(pair_result["pair_id"])
                if source_pair is None:
                    continue
                cell = next((key for key, prediction in reusable_predictions.items()
                             if prediction["manifest"]["producer"] == source_pair["prediction_producer"]
                             and prediction["manifest"]["fingerprint"] == source_pair["prediction_fingerprint"]
                             and prediction["manifest"]["metadata"].get("task") == pair_result["task"]
                             and source_pair["annotation_fingerprint"] == prediction["annotation_fingerprint"]), None)
                if cell is None or cell in selected:
                    continue
                roles = {"judge.raw.jsonl": "judge-raw-", "judge.scored.jsonl": "judge-scored-",
                         "judge.metrics.json": "pair-metrics-"}
                artifacts = {}
                for filename, prefix in roles.items():
                    role = prefix + pair_result["pair_id"]
                    manifest = info["outputs"].get(role)
                    if manifest is None:
                        raise ValueError("Audited source eval lacks pair artifact: " + role)
                    artifacts[filename] = {"path": manifest["path"], "sha256": sha256_file(manifest["path"]),
                                           "manifest_fingerprint": manifest["fingerprint"]}
                source_metrics = read_json(artifacts["judge.metrics.json"]["path"])
                selected[cell] = {
                    "source": {"experiment_id": experiment_id, "run_id": run["run_id"],
                               "job_id": job["job_id"], "setting": exp_snapshot["config"]["experiment"]["setting"]},
                    "source_pair_id": pair_result["pair_id"], "source_pair": source_pair,
                    "source_request_path": str(request_path), "source_request_sha256": request_sha,
                    "run": source_metrics["run"], "run_fingerprint": source_metrics["run_fingerprint"],
                    "eval_params": request["params"], "judge_model_fingerprint": request["judge_model_fingerprint"],
                    "judge_processor_fingerprint": request["judge_processor_fingerprint"],
                    "judge_source_sha256": request["judge_source_sha256"],
                    "prediction_fingerprint": source_pair["prediction_fingerprint"],
                    "annotation_fingerprint": source_pair["annotation_fingerprint"],
                    "artifacts": artifacts,
                }
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return selected


def audit_reusable_prediction(snapshot, task, candidate, manifest, expected_inputs, source):
    """Recheck a frozen imported prediction against its source job and exact baseline inputs."""
    if manifest != candidate.get("manifest"):
        raise ValueError("Imported prediction manifest differs from the frozen reuse candidate")
    for role, expected_key in (("base-model", "base_model_fingerprint"),
                               ("processor", "processor_fingerprint")):
        if expected_inputs[role]["fingerprint"] != candidate.get(expected_key):
            raise ValueError("Imported prediction {} identity changed".format(role))
    if candidate.get("annotation_fingerprint") != expected_inputs["annotation-{}-test".format(task)]["fingerprint"]:
        raise ValueError("Imported prediction annotation identity changed")
    info = _infer_source_info(snapshot, candidate, source, task)
    if (info["manifest"] != manifest
            or info["record"]["inputs"]["adapter"]["identity"] != source["adapters"][candidate["source_task"]]["manifest"]["identity"]):
        raise ValueError("Imported prediction artifact or adapter changed after baseline preparation")
    return info


def audit_reusable_evaluation(snapshot, pair, candidate):
    """Recheck frozen source judge artifacts against the exact prediction and judge identity."""
    from .common import sha256_file

    experiment = snapshot["config"]["experiment"]
    if (candidate.get("eval_params") != snapshot["config"]["eval"]["params"]
            or candidate.get("source_pair", {}).get("prediction_sha256") != pair["prediction_sha256"]
            or candidate.get("annotation_fingerprint") != pair["annotation_fingerprint"]
            or candidate.get("source_pair", {}).get("annotation_fingerprint") != pair["annotation_fingerprint"]):
        raise ValueError("Reusable judge result does not match the baseline prediction, GT or eval params")
    judge_cache = ManifestVerificationCache()
    judge = make_manifest("judge_model", experiment["judge_model"],
                          source="configuration:experiment.judge_model", verification_cache=judge_cache)
    processor = make_manifest("processor", experiment["judge_model"],
                              source="configuration:experiment.judge_model", verification_cache=judge_cache)
    if (candidate.get("judge_model_fingerprint") != judge["fingerprint"]
            or candidate.get("judge_processor_fingerprint") != processor["fingerprint"]):
        raise ValueError("Reusable judge model or processor identity differs")
    code_dir = Path(experiment["repo_root"]) / "evaluation/COSINE"
    code_hashes = {name: sha256_file(code_dir / name)
                   for name in ("eval_cosine.py", "dataset_utils.py", "qwen_judge.py")}
    if candidate.get("judge_source_sha256") != code_hashes:
        raise ValueError("Reusable judge source code differs")
    request_path = Path(candidate["source_request_path"])
    if sha256_file(request_path) != candidate["source_request_sha256"]:
        raise ValueError("Reusable judge request changed after discovery")
    request = read_json(request_path)
    pair_id_value = candidate["source_pair_id"]
    source_pair = next((item for item in request.get("pairs", []) if item.get("pair_id") == pair_id_value), None)
    if (source_pair != candidate["source_pair"]
            or request.get("params") != snapshot["config"]["eval"]["params"]
            or request.get("judge_model_fingerprint") != judge["fingerprint"]
            or request.get("judge_processor_fingerprint") != processor["fingerprint"]
            or request.get("judge_source_sha256") != code_hashes
            or source_pair is None
            or source_pair.get("prediction_sha256") != pair["prediction_sha256"]
            or source_pair.get("annotation_fingerprint") != pair["annotation_fingerprint"]):
        raise ValueError("Reusable judge request does not match its frozen pair and judge identity")
    for filename, item in candidate["artifacts"].items():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Reusable judge artifact changed: " + filename)
    source_metrics = read_json(candidate["artifacts"]["judge.metrics.json"]["path"])
    if (source_metrics.get("run") != candidate["run"]
            or source_metrics.get("run_fingerprint") != candidate["run_fingerprint"]
            or source_metrics.get("combined_flat_rawacc") is None):
        raise ValueError("Reusable judge metrics differ from the frozen source run")
    return True
