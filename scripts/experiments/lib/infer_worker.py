"""One vLLM process for every test task assigned to one Adapter/base job.

Keep top-level imports free of model libraries: the executor also uses the
annotation and prediction inspection helpers without importing CUDA.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

if __package__ in (None, ""):
    from common import TASKS, fingerprint, read_json, sha256_file, utc_now, write_json_atomic
else:
    from .common import TASKS, fingerprint, read_json, sha256_file, utc_now, write_json_atomic


def load_samples(annotation_path, media_root):
    """Read the frozen annotation path; never consult the dataset registry."""
    annotation_path = Path(annotation_path)
    if annotation_path.suffix.lower() == ".jsonl":
        with annotation_path.open(encoding="utf-8") as handle:
            annotations = [json.loads(line) for line in handle if line.strip()]
    else:
        annotations = read_json(annotation_path)
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("A frozen test annotation must contain a nonempty list")
    samples = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict) or not isinstance(annotation.get("_meta", {}), dict):
            raise ValueError("Invalid test annotation object at index {}".format(index))
        conversations = annotation.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("Test conversations must be a list at index {}".format(index))
        answers = [turn.get("value") for turn in conversations
                   if isinstance(turn, dict) and turn.get("from") == "gpt"]
        if len(answers) != 1 or not isinstance(answers[0], str):
            raise ValueError("Each test sample requires one textual gpt reference answer")
        sample = dict(annotation)
        sample["data_path"] = str(media_root)
        samples.append(sample)
    return samples


def inspect_predictions(task_spec, samples, params):
    """Require a complete, ordered legacy-format prediction file for one task."""
    output = Path(task_spec["output_path"])
    generation = {"max_new_tokens": params["max_new_tokens"], "do_sample": params["do_sample"],
                  "temperature": params["temperature"] if params["do_sample"] else None,
                  "top_p": params["top_p"] if params["do_sample"] else None}
    count = empty = 0
    with output.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= len(samples):
                raise ValueError("Predictions contain more rows than the frozen annotation")
            record = json.loads(line)
            sample = samples[index]
            meta = sample.get("_meta", {})
            answer = next(turn["value"] for turn in sample["conversations"]
                          if isinstance(turn, dict) and turn.get("from") == "gpt")
            expected = {"dataset": task_spec["dataset"], "inference_index": index,
                        "sample_index": meta.get("sample_index", index), "id": meta.get("id"),
                        "task_type": meta.get("task_type"), "metric_type": meta.get("metric_type"),
                        "_meta": meta, "ground_truth": answer, "generation": generation}
            if (not isinstance(record, dict) or not set(expected).issubset(record)
                    or type(record.get("inference_index")) is not int
                    or any(record[key] != value for key, value in expected.items())
                    or not isinstance(record.get("prediction"), str)):
                raise ValueError("Prediction metadata/order mismatch at {} index {}".format(
                    task_spec["task"], index))
            count += 1
            empty += not record["prediction"].strip()
    if count != len(samples):
        raise ValueError("Predictions do not cover the complete frozen test annotation")
    return {"task": task_spec["task"], "dataset": task_spec["dataset"],
            "output_path": str(output), "sample_count": count,
            "empty_predictions": empty, "sha256": sha256_file(output)}


def _request(path):
    request = read_json(path)
    payload = {key: value for key, value in request.items() if key != "request_fingerprint"}
    if request.get("schema_version") != 1 or request.get("request_fingerprint") != fingerprint(payload):
        raise ValueError("Inference request fingerprint mismatch")
    group = request["physical_gpu_ids"]
    if (not isinstance(group, list) or not group or len(set(group)) != len(group)
            or any(type(index) is not int or index not in (1, 2, 3) for index in group)
            or len(group) != request["params"]["tensor_parallel_size"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, group))
            or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"):
        raise ValueError("Worker must inherit the explicitly assigned physical GPUs 1-3")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    rows = {int(parts[0]): parts[1:] for line in result.stdout.splitlines() if line.strip()
            for parts in [[part.strip() for part in line.split(",")]]}
    identity = {"physical_ids": group, "uuids": [rows[index][0] for index in group],
                "pci_bus_ids": [rows[index][1] for index in group]}
    if identity != request["gpu_identity"]:
        raise ValueError("Physical GPU identity changed before the worker started")
    output = Path(request["output_dir"])
    if not output.is_absolute() or not output.is_dir():
        raise ValueError("Worker requires the executor-created absolute output directory")
    tasks = request["tasks"]
    if (not isinstance(tasks, list) or not tasks
            or any(item["task"] not in TASKS for item in tasks)
            or len({item["task"] for item in tasks}) != len(tasks)):
        raise ValueError("Worker requires distinct planned test tasks")
    for item in tasks:
        if (item["dataset"] != "cosine_{}_test".format(item["task"])
                or Path(item["output_path"]) != output / "predictions" / (item["task"] + ".jsonl")
                or not Path(item["annotation_path"]).is_absolute()
                or sha256_file(item["annotation_path"]) != item["annotation_sha256"]):
            raise ValueError("Test task paths or annotation contents disagree with the request")
        if Path(item["output_path"]).exists() or Path(item["output_path"]).is_symlink():
            raise FileExistsError("Refusing to overwrite predictions: " + item["output_path"])
    if (output / "inference_result.json").exists():
        raise FileExistsError("An inference result already exists")
    return request


def _configure_processor(processor, params):
    """Apply frozen image/video sampling before saving vLLM's processor view."""
    image = processor.image_processor
    video = processor.video_processor
    if not isinstance(image.size, dict) or not isinstance(video.size, dict):
        raise ValueError("Expected Qwen3-VL image and video pixel-size dictionaries")
    image.size = {**image.size, "shortest_edge": params["min_pixels"],
                  "longest_edge": params["max_pixels"]}
    video.size = {**video.size, "shortest_edge": params["video_min_pixels"],
                  "longest_edge": params["video_max_pixels"]}
    for processor_part, minimum, maximum in (
            (image, params["min_pixels"], params["max_pixels"]),
            (video, params["video_min_pixels"], params["video_max_pixels"])):
        if hasattr(processor_part, "min_pixels"):
            processor_part.min_pixels = minimum
        if hasattr(processor_part, "max_pixels"):
            processor_part.max_pixels = maximum
    settings = {"min_frames": params["video_min_frames"], "max_frames": params["video_max_frames"],
                "fps": params["video_fps"], "num_frames": None, "do_sample_frames": True}
    for key, value in settings.items():
        if not hasattr(video, key):
            raise ValueError("The selected video processor does not support " + key)
        setattr(video, key, value)
    return {"image_size": dict(image.size), "video_size": dict(video.size), **settings}


def execute_worker(request_path):
    request = _request(request_path)
    output = Path(request["output_dir"])
    started = time.monotonic()
    summary = {"schema_version": 1, "status": "running", "started_at": utc_now(),
               "producer": request["producer"], "config_fingerprint": request["config_fingerprint"],
               "request_fingerprint": request["request_fingerprint"],
               "input_fingerprints": request["input_fingerprints"], "params": request["params"],
               "physical_gpu_ids": request["physical_gpu_ids"], "gpu_identity": request["gpu_identity"],
               "task_order": [item["task"] for item in request["tasks"]],
               "model_loads": 0, "engine_shutdown": False, "tasks": []}
    try:
        # Visibility and identity have already been checked without CUDA imports.
        source = Path(request["repo_root"]) / "qwen-vl-finetune"
        sys.path.insert(0, str(source))
        from qwenvl.infer import infer_cosine as legacy

        args = argparse.Namespace(**request["params"], base_model=request["base_model"],
                                  processor_path=request["processor_path"],
                                  adapter_path=request["adapter_path"], overwrite=False)
        legacy.validate_runtime(args)
        datasets = []
        limits = {"image": 0, "video": 0}
        for item in request["tasks"]:
            samples = load_samples(item["annotation_path"], request["media_root"])
            task_limits = legacy.preflight_samples(samples)
            for modality in limits:
                limits[modality] = max(limits[modality], task_limits[modality])
            datasets.append((item, samples))
        processor = legacy.load_processor(args)
        summary["processor_sampling"] = _configure_processor(processor, request["params"])
        adapter_config = legacy.load_adapter_config(args)
        summary["media_limits"] = limits
        with TemporaryDirectory(prefix="pipeline-vllm-model-") as directory:
            model_directory = Path(directory)
            legacy.prepare_model_directory(args, processor, model_directory)
            engine, sampling, lora_request = legacy.load_engine(args, model_directory, limits, adapter_config)
            summary["model_loads"] = 1
            inference_error = None
            try:
                for item, samples in datasets:
                    print("Running {} ({} samples) with the shared engine".format(
                        item["dataset"], len(samples)), flush=True)
                    legacy.run_inference(engine, processor, samples, item["dataset"],
                                         Path(item["output_path"]), args, sampling, lora_request)
                    summary["tasks"].append(inspect_predictions(item, samples, request["params"]))
            except BaseException as error:
                inference_error = error
                raise
            finally:
                try:
                    legacy.shutdown_vllm_engine(engine)
                    summary["engine_shutdown"] = True
                except Exception as error:
                    summary["shutdown_error"] = str(error)
                    if inference_error is None:
                        raise
                    print("vLLM shutdown failed: " + str(error), file=sys.stderr, flush=True)
                finally:
                    engine = None
                    gc.collect()
        for item in request["tasks"]:
            if sha256_file(item["annotation_path"]) != item["annotation_sha256"]:
                raise ValueError("An annotation changed during inference")
        summary.update(status="succeeded", finished_at=utc_now(),
                       elapsed_seconds=time.monotonic() - started)
        write_json_atomic(output / "inference_result.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                       finished_at=utc_now(), elapsed_seconds=time.monotonic() - started,
                       error={"type": type(error).__name__, "message": str(error)})
        write_json_atomic(output / "inference_failure.json", summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Absolute frozen inference.request.json path")
    args = parser.parse_args(argv)
    if not Path(args.request).is_absolute():
        parser.error("--request must be an absolute path")
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Inference worker received signal {}".format(signum))
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
