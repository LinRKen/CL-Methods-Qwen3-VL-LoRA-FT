"""Judge all frozen prediction/task pairs with one text-only Qwen engine.

Model libraries are imported only after physical GPU identity verification.
Inspection helpers remain usable without CUDA or a model load.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    from common import TASKS, fingerprint, read_json, require_id, sha256_file, utc_now, write_json_atomic
    from infer_worker import inspect_predictions, load_samples
else:
    from .common import TASKS, fingerprint, read_json, require_id, sha256_file, utc_now, write_json_atomic
    from .infer_worker import inspect_predictions, load_samples


def pair_id(pair):
    return require_id(pair["prediction_job"] + "--" + pair["task"])


def _legacy(repo_root):
    directory = (Path(repo_root) / "evaluation/COSINE").resolve(strict=True)
    sys.path.insert(0, str(directory))
    try:
        for name in ("dataset_utils", "qwen_judge", "eval_cosine"):
            module = importlib.import_module(name)
            if Path(module.__file__).resolve() != directory / (name + ".py"):
                raise ValueError("An unrelated module shadows the COSINE judge: " + name)
        return sys.modules["eval_cosine"]
    finally:
        sys.path.remove(str(directory))


def _check_sources(request):
    directory = Path(request["repo_root"]) / "evaluation/COSINE"
    actual = {name: sha256_file(directory / name)
              for name in ("eval_cosine.py", "dataset_utils.py", "qwen_judge.py")}
    if actual != request["judge_source_sha256"]:
        raise ValueError("Judge source changed after the worker request was frozen")


def prepare_pair(pair, repo_root):
    """Align complete predictions with the exact frozen GT, without registry lookup."""
    for kind in ("prediction", "annotation"):
        path = Path(pair[kind + "_path"])
        if not path.is_absolute() or sha256_file(path) != pair[kind + "_sha256"]:
            raise ValueError("Frozen pair input changed: " + str(path))
    samples = load_samples(pair["annotation_path"], Path(pair["annotation_path"]).parent.parent)
    inspect_predictions({"task": pair["task"], "dataset": pair["dataset"],
                         "output_path": pair["prediction_path"]}, samples, pair["infer_params"])
    legacy = _legacy(repo_root)
    records = legacy.align_predictions(legacy.read_jsonl(pair["prediction_path"]), samples, pair["dataset"])
    for kind in ("prediction", "annotation"):
        if sha256_file(pair[kind + "_path"]) != pair[kind + "_sha256"]:
            raise ValueError("Pair input changed while it was read")
    return records


def _pipeline_identity(request, pair):
    return {"request_fingerprint": request["request_fingerprint"],
            "producer": request["producer"], "config_fingerprint": request["config_fingerprint"],
            "pair_id": pair["pair_id"], "prediction_fingerprint": pair["prediction_fingerprint"],
            "annotation_fingerprint": pair["annotation_fingerprint"],
            "judge_model_fingerprint": request["judge_model_fingerprint"],
            "judge_processor_fingerprint": request["judge_processor_fingerprint"]}


def _run_fingerprint(run):
    # Preserve the existing COSINE raw-file fingerprint convention.
    packed = json.dumps(run, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _make_run(legacy, request, pair, args):
    run, _ = legacy.make_run(args, Path(pair["prediction_path"]),
                             Path(pair["annotation_path"]), Path(request["judge_model"]))
    run["pipeline"] = _pipeline_identity(request, pair)
    return run, _run_fingerprint(run)


def _validate_run(legacy, request, pair, run):
    expected = {"schema_version": 1, "dataset": pair["dataset"],
                "input_file": pair["prediction_path"], "input_sha256": pair["prediction_sha256"],
                "annotation_file": pair["annotation_path"], "annotation_sha256": pair["annotation_sha256"],
                "judge_model": legacy.model_identity(Path(request["judge_model"])),
                "prompt_version": legacy.PROMPT_VERSION, "source_sha256": request["judge_source_sha256"],
                "settings": request["params"], "sampling": {"temperature": 0.0, "top_p": 1.0, "n": 1},
                "pipeline": _pipeline_identity(request, pair)}
    if (not isinstance(run, dict) or set(run) != set(expected) | {"packages"}
            or any(fingerprint(run.get(key)) != fingerprint(value) for key, value in expected.items())
            or not isinstance(run["packages"], dict)
            or set(run["packages"]) != {"vllm", "transformers", "torch"}
            or any(value is not None and not isinstance(value, str) for value in run["packages"].values())):
        raise ValueError("Judge raw run does not match its frozen pair and request")


def _devices(request):
    identity = request["gpu_identity"]
    return [{"physical_index": physical, "uuid": uuid, "pci_bus_id": pci}
            for physical, uuid, pci in zip(request["physical_gpu_ids"], identity["uuids"], identity["pci_bus_ids"])]


def inspect_pair(request, pair):
    """Reparse raw judgments and recompute scores; never call the judge model."""
    legacy = _legacy(request["repo_root"])
    records = prepare_pair(pair, request["repo_root"])
    directory = Path(pair["output_dir"])
    raw_path, scored_path, metrics_path = (directory / name for name in legacy.ARTIFACTS)
    metrics = read_json(metrics_path)
    run = metrics.get("run")
    _validate_run(legacy, request, pair, run)
    run_fingerprint = _run_fingerprint(run)
    done = legacy.restore_raw(raw_path, records, run, run_fingerprint)
    if set(done) != set(range(len(records))):
        raise ValueError("Raw judge output is missing planned samples: " + pair["pair_id"])
    if any(row.get("physical_gpus") != _devices(request)
           or not isinstance(row.get("completed_at"), str) or not row["completed_at"] for row in done.values()):
        raise ValueError("Raw judge device or completion evidence is missing")
    scored = [legacy.scored_record(record, done[record["inference_index"]]["parsed"],
                                  done[record["inference_index"]]["parse_error"]) for record in records]
    if fingerprint(legacy.read_jsonl(scored_path)) != fingerprint(scored):
        raise ValueError("Scored rows differ from the complete raw judgments")
    expected = legacy.aggregate_metrics(scored)
    expected.update(run=run, run_fingerprint=run_fingerprint,
                    judge_parse_failures=sum(row["judge_parse_error"] is not None for row in scored),
                    raw_file=str(raw_path), scored_file=str(scored_path))
    if (set(metrics) != set(expected) | {"completed_at"}
            or not isinstance(metrics["completed_at"], str) or not metrics["completed_at"]
            or any(fingerprint(metrics.get(key)) != fingerprint(value) for key, value in expected.items())):
        raise ValueError("Pair metrics differ from raw/scored recomputation")
    result = {key: pair[key] for key in ("pair_id", "prediction_job", "prediction_slot", "task", "dataset",
              "prediction_fingerprint", "annotation_fingerprint")}
    result.update({key: metrics[key] for key in ("total", "valid", "invalid", "score_sum",
                   "combined_flat_rawacc", "judge_parse_failures")})
    result.update(output_dir=str(directory), run_fingerprint=run_fingerprint,
                  artifacts={name: {"path": str(directory / name), "sha256": sha256_file(directory / name)}
                             for name in legacy.ARTIFACTS})
    return result, scored


def _combined(request, results, scored):
    legacy = _legacy(request["repo_root"])
    metrics = legacy.aggregate_metrics(scored)
    metrics.update(schema_version=1, producer=request["producer"],
                   config_fingerprint=request["config_fingerprint"],
                   request_fingerprint=request["request_fingerprint"],
                   input_fingerprints=request["input_fingerprints"], params=request["params"],
                   judge_model_fingerprint=request["judge_model_fingerprint"],
                   judge_parse_failures=sum(row["judge_parse_error"] is not None for row in scored),
                   pair_order=[pair["pair_id"] for pair in request["pairs"]], pairs=results,
                   aggregation_scope="All prediction/sample pairs; repeated tests across adapters remain separate.")
    return metrics


def inspect_evaluation(request):
    _check_sources(request)
    results, scored = [], []
    for pair in request["pairs"]:
        result, rows = inspect_pair(request, pair)
        results.append(result)
        scored.extend(rows)
    expected = _combined(request, results, scored)
    output = Path(request["output_dir"])
    if fingerprint(read_json(output / "judge.metrics.json")) != fingerprint(expected):
        raise ValueError("Combined metrics differ from the complete ordered pair results")
    summary = read_json(output / "evaluation_result.json")
    fields = {**expected, "status": "succeeded", "physical_gpu_ids": request["physical_gpu_ids"],
              "gpu_identity": request["gpu_identity"], "model_loads": 1, "engine_shutdown": True,
              "metrics_file": str(output / "judge.metrics.json"),
              "metrics_sha256": sha256_file(output / "judge.metrics.json")}
    if (set(summary) != set(fields) | {"started_at", "finished_at", "elapsed_seconds"}
            or any(fingerprint(summary.get(key)) != fingerprint(value) for key, value in fields.items())
            or not isinstance(summary["started_at"], str) or not isinstance(summary["finished_at"], str)
            or type(summary["elapsed_seconds"]) not in (int, float) or summary["elapsed_seconds"] < 0):
        raise ValueError("Evaluation summary is incomplete or belongs to another execution")
    return summary


def _request(path):
    request = read_json(path)
    payload = {key: value for key, value in request.items() if key != "request_fingerprint"}
    if request.get("schema_version") != 1 or request.get("request_fingerprint") != fingerprint(payload):
        raise ValueError("Evaluation request fingerprint mismatch")
    group = request["physical_gpu_ids"]
    if (not isinstance(group, list) or not group or len(set(group)) != len(group)
            or any(type(index) is not int or index not in (1, 2, 3) for index in group)
            or len(group) != request["params"]["tensor_parallel_size"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, group))
            or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"):
        raise ValueError("Worker must inherit explicitly assigned physical GPUs 1-3")
    _check_sources(request)
    legacy = _legacy(request["repo_root"])
    args = argparse.Namespace(**request["params"])
    if legacy.verify_gpu_selection(args) != _devices(request):
        raise ValueError("Physical GPU identity changed before judge startup")
    output = Path(request["output_dir"])
    if not output.is_absolute() or not output.is_dir() or output.is_symlink():
        raise ValueError("Worker requires the executor-created output directory")
    pairs = request["pairs"]
    if (not isinstance(pairs, list) or not pairs
            or len({pair["pair_id"] for pair in pairs}) != len(pairs)
            or len(pairs) != len(request["lineage"]["scoring_pairs"])):
        raise ValueError("Worker requires distinct complete scoring pairs")
    for pair, planned in zip(pairs, request["lineage"]["scoring_pairs"]):
        if (any(pair.get(key) != value for key, value in planned.items())
                or pair["task"] not in TASKS or pair["pair_id"] != pair_id(pair)
                or pair["dataset"] != "cosine_{}_test".format(pair["task"])
                or Path(pair["output_dir"]) != output / "pairs" / pair["pair_id"]
                or request["input_fingerprints"].get(pair["prediction_role"]) != pair["prediction_fingerprint"]
                or request["input_fingerprints"].get("annotation-{}-test".format(pair["task"])) != pair["annotation_fingerprint"]):
            raise ValueError("Pair paths or bindings differ from the frozen plan")
        if Path(pair["output_dir"]).exists() or Path(pair["output_dir"]).is_symlink():
            raise FileExistsError("Refusing to overwrite a scoring pair output")
    for name in ("judge.metrics.json", "evaluation_result.json", "evaluation_failure.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError(output / name)
    return request


def execute_worker(request_path):
    request = _request(request_path)
    output = Path(request["output_dir"])
    started = time.monotonic()
    summary = {"schema_version": 1, "status": "running", "started_at": utc_now(),
               "producer": request["producer"], "request_fingerprint": request["request_fingerprint"],
               "model_loads": 0, "engine_shutdown": False, "pairs": []}
    try:
        legacy = _legacy(request["repo_root"])
        model = Path(request["judge_model"])
        processor = legacy.load_processor(model)
        prepared = []
        # Preflight every GT alignment and prompt before allocating the model engine.
        for pair in request["pairs"]:
            records = prepare_pair(pair, request["repo_root"])
            args = argparse.Namespace(**request["params"], dataset=pair["dataset"])
            run, run_fingerprint = _make_run(legacy, request, pair, args)
            prompts = {row["inference_index"]: legacy.make_prompt(processor, legacy.build_messages(row),
                       args, row["inference_index"]) for row in records}
            prepared.append((pair, records, args, run, run_fingerprint, prompts))
        engine, sampling = legacy.load_engine(argparse.Namespace(**request["params"]), model)
        summary["model_loads"] = 1
        evaluation_error = None
        try:
            for pair, records, args, run, run_fingerprint, prompts in prepared:
                directory = Path(pair["output_dir"])
                directory.mkdir(parents=True, exist_ok=False)
                raw_path, scored_path, metrics_path = (directory / name for name in legacy.ARTIFACTS)
                done = {}
                print("Judging {} ({} samples) with the shared engine".format(pair["pair_id"], len(records)), flush=True)
                with raw_path.open("x", encoding="utf-8") as handle:
                    for offset in range(0, len(records), args.batch_size):
                        legacy.judge_batch(engine, sampling, processor, records[offset:offset + args.batch_size],
                                           prompts, args, handle, done, run, run_fingerprint, _devices(request), len(records))
                if set(done) != set(range(len(records))):
                    raise ValueError("Judge did not produce every planned sample")
                scored = [legacy.scored_record(row, done[row["inference_index"]]["parsed"],
                          done[row["inference_index"]]["parse_error"]) for row in records]
                metrics = legacy.aggregate_metrics(scored)
                metrics.update(run=run, run_fingerprint=run_fingerprint, completed_at=utc_now(),
                               judge_parse_failures=sum(row["judge_parse_error"] is not None for row in scored),
                               raw_file=str(raw_path), scored_file=str(scored_path))
                legacy.write_json_atomic(scored_path, scored, jsonl=True)
                write_json_atomic(metrics_path, metrics)
                summary["pairs"].append({"pair_id": pair["pair_id"], "total": len(scored),
                                         "combined_flat_rawacc": metrics["combined_flat_rawacc"]})
        except BaseException as error:
            evaluation_error = error
            raise
        finally:
            try:
                legacy.shutdown_engine(engine)
                summary["engine_shutdown"] = True
            except Exception as error:
                summary["shutdown_error"] = str(error)
                if evaluation_error is None:
                    raise
                print("vLLM shutdown failed: " + str(error), file=sys.stderr, flush=True)
            finally:
                engine = None
                gc.collect()
        _check_sources(request)
        results, all_scored = [], []
        for pair in request["pairs"]:
            result, scored = inspect_pair(request, pair)
            results.append(result)
            all_scored.extend(scored)
        combined = _combined(request, results, all_scored)
        write_json_atomic(output / "judge.metrics.json", combined)
        summary.update(combined)
        summary.update(status="succeeded", physical_gpu_ids=request["physical_gpu_ids"],
                       gpu_identity=request["gpu_identity"], metrics_file=str(output / "judge.metrics.json"),
                       metrics_sha256=sha256_file(output / "judge.metrics.json"),
                       finished_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json_atomic(output / "evaluation_result.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                       finished_at=utc_now(), elapsed_seconds=time.monotonic() - started,
                       error={"type": type(error).__name__, "message": str(error)})
        write_json_atomic(output / "evaluation_failure.json", summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Absolute frozen evaluation.request.json path")
    args = parser.parse_args(argv)
    if not Path(args.request).is_absolute():
        parser.error("--request must be an absolute path")
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Evaluation worker received signal {}".format(signum))
    previous = {number: signal.signal(number, interrupt) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        execute_worker(args.request)
    except (KeyboardInterrupt, InterruptedError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 130
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
