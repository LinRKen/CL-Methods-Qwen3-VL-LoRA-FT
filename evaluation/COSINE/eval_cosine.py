#!/usr/bin/env python3
"""Score an existing complete COSINE infer JSONL with a local Qwen judge."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import subprocess

from dataset_utils import TEST_DATASETS, align_predictions, load_registered_test, read_jsonl, sha256_file, write_json_atomic
from qwen_judge import PROMPT_VERSION, aggregate_metrics, build_messages, build_repair_messages, parse_judge_response, scored_record

DEFAULT_MODEL = "/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct"
ARTIFACTS = ("judge.raw.jsonl", "judge.scored.jsonl", "judge.metrics.json")
ALLOWED_GPUS = {"1", "2", "3"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=TEST_DATASETS)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "tensor_parallel_size", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.max_model_len <= args.max_new_tokens or args.repair_attempts < 0:
        parser.error("max-model-len must exceed max-new-tokens and repair-attempts cannot be negative")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def verify_gpu_selection(args):
    """Check physical GPU identity before importing transformers, torch, or vLLM."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = [item.strip() for item in visible.split(",")]
    if (not visible or len(set(selected)) != len(selected)
            or any(item not in ALLOWED_GPUS and not item.startswith("GPU-") for item in selected)):
        raise RuntimeError("Explicitly select physical GPUs 1, 2, 3 or full UUIDs; physical GPU 0 is forbidden")
    if len(selected) != args.tensor_parallel_size:
        raise ValueError("--tensor-parallel-size must match the number of selected GPUs")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot verify physical GPUs; refusing to initialize the judge") from error
    rows = [tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    if (not rows or any(len(row) != 3 for row in rows)
            or len({row[0] for row in rows}) != len(rows)
            or len({row[1] for row in rows}) != len(rows)):
        raise RuntimeError("Invalid physical GPU identity query result")
    try:
        pci_order = sorted(rows, key=lambda row: tuple(
            int(part, 16) for part in row[2].replace(".", ":").split(":")))
    except ValueError as error:
        raise RuntimeError("Cannot parse GPU PCI bus identities") from error
    if [row[0] for row in pci_order] != [str(index) for index in range(len(rows))]:
        raise RuntimeError("CUDA PCI order differs from physical NVML indices; refusing to start")
    allowed = {index: (uuid, pci) for index, uuid, pci in rows if index in ALLOWED_GPUS}
    by_uuid = {uuid: index for index, (uuid, _) in allowed.items()}
    indices = [item if item in allowed else by_uuid.get(item) for item in selected]
    if None in indices or len(set(indices)) != len(indices):
        raise RuntimeError("Selected physical GPU is missing, duplicated, or forbidden")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    devices = [{"physical_index": int(index), "uuid": allowed[index][0],
                "pci_bus_id": allowed[index][1]} for index in indices]
    print("Verified physical GPUs: {}".format(devices), flush=True)
    return devices


def model_identity(path):
    if not path.is_dir():
        raise NotADirectoryError(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("--judge-model must point to a local Qwen3-VL model")
    weights = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError("No local model weight files found: {}".format(path))
    files = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "chat_template.json", "chat_template.jinja", "processor_config.json",
             "preprocessor_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "merges.txt", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    return {"path": str(path),
            "config_sha256": {name: sha256_file(path / name) for name in files if (path / name).is_file()},
            "weight_files": [{"name": item.name, "size": item.stat().st_size,
                              "mtime_ns": item.stat().st_mtime_ns} for item in weights],
            "weight_identity_method": "filename/size/mtime; weight contents are not fully hashed"}


def make_run(args, input_path, annotation_path, model_path):
    packages = {}
    for name in ("vllm", "transformers", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).resolve().parent
    run = {"schema_version": 1, "dataset": args.dataset,
           "input_file": str(input_path), "input_sha256": sha256_file(input_path),
           "annotation_file": str(annotation_path), "annotation_sha256": sha256_file(annotation_path),
           "judge_model": model_identity(model_path), "prompt_version": PROMPT_VERSION,
           "source_sha256": {name: sha256_file(source / name)
                             for name in ("eval_cosine.py", "dataset_utils.py", "qwen_judge.py")},
           "packages": packages,
           "settings": {name: getattr(args, name) for name in (
               "batch_size", "tensor_parallel_size", "gpu_memory_utilization", "max_model_len",
               "max_new_tokens", "repair_attempts", "seed", "dtype")},
           "sampling": {"temperature": 0.0, "top_p": 1.0, "n": 1}}
    packed = json.dumps(run, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return run, hashlib.sha256(packed).hexdigest()


def prepare_artifacts(args, input_path, annotation_path):
    output = Path(args.output_dir).expanduser().resolve()
    paths = tuple(output / name for name in ARTIFACTS)
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Artifact must be a regular file: {}".format(path))
        if path.resolve() in (input_path, annotation_path):
            raise ValueError("Refusing to overwrite input data: {}".format(path))
    existing = [path for path in paths if path.exists()]
    if existing and not (args.resume or args.overwrite):
        raise FileExistsError("Output exists; specify --resume or --overwrite")
    if args.resume and existing and not paths[0].exists():
        raise FileNotFoundError("Cannot resume without judge.raw.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()
    return paths


def restore_raw(path, records, run, fingerprint):
    done = {}
    if not path.exists():
        return done
    for raw in read_jsonl(path):
        if raw.get("run_fingerprint") != fingerprint or raw.get("run") != run:
            raise ValueError("Resume run configuration does not match raw results")
        record = raw.get("record")
        index = record.get("inference_index") if isinstance(record, dict) else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records) or index in done:
            raise ValueError("Invalid or duplicate sample index in resume file")
        if record != records[index]:
            raise ValueError("Resume sample no longer matches inference input at index {}".format(index))
        attempts = raw.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= run["settings"]["repair_attempts"] + 1:
            raise ValueError("Invalid attempt history in resume file")
        messages, parsed, error_text = build_messages(record), None, None
        for number, attempt in enumerate(attempts):
            if (not isinstance(attempt, dict) or attempt.get("messages") != messages
                    or not isinstance(attempt.get("response"), str)
                    or attempt.get("finish_reason") not in ("stop", "length")):
                raise ValueError("Invalid recorded judge attempt at sample {}".format(index))
            try:
                parsed = parse_judge_response(attempt["response"], record["metric_type"])
                error_text = None
            except (ValueError, TypeError) as error:
                parsed, error_text = None, str(error)
            if attempt.get("parse_error") != error_text:
                raise ValueError("Recorded parse status is inconsistent at sample {}".format(index))
            if number < len(attempts) - 1:
                if error_text is None:
                    raise ValueError("Unexpected repair after valid judge output")
                messages = build_repair_messages(messages, attempt["response"], error_text)
        if (parsed != raw.get("parsed") or error_text != raw.get("parse_error")
                or (error_text is not None and len(attempts) != run["settings"]["repair_attempts"] + 1)):
            raise ValueError("Incomplete raw row at sample {}".format(index))
        done[index] = raw
    return done


def make_prompt(processor, messages, args, index):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(tokens) + args.max_new_tokens > args.max_model_len:
        raise ValueError("Sample {}: prompt plus output budget exceeds max-model-len; no truncation".format(index))
    return prompt


def load_processor(model_path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    if processor.__class__.__name__ != "Qwen3VLProcessor":
        raise ValueError("A local Qwen3-VL processor is required")
    return processor


def load_engine(args, model_path):
    from importlib.metadata import version as package_version
    from packaging.version import Version
    if Version(package_version("vllm")) < Version("0.11.0"):
        raise RuntimeError("Evaluation requires vLLM >= 0.11.0")
    from vllm import LLM, SamplingParams
    settings = dict(model=str(model_path), tokenizer=str(model_path), dtype=args.dtype,
                    tensor_parallel_size=args.tensor_parallel_size, max_num_seqs=args.batch_size,
                    gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                    seed=args.seed, limit_mm_per_prompt={"image": 0, "video": 0}, generation_config="vllm")
    if args.tensor_parallel_size > 1:
        settings["distributed_executor_backend"] = "mp"
    engine = LLM(**settings)
    sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                              seed=args.seed, skip_special_tokens=True)
    return engine, sampling


def shutdown_engine(engine):
    core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("This vLLM version does not expose engine-core shutdown")
    shutdown()


def append_raw(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def judge_batch(engine, sampling, processor, batch, prompts, args, handle, done, run, fingerprint, devices, total):
    active = [{"record": record, "messages": build_messages(record), "attempts": [],
               "prompt": prompts[record["inference_index"]]}
              for record in batch]
    while active:
        outputs = engine.generate([item["prompt"] for item in active], sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(active):
            raise RuntimeError("vLLM returned a different number of outputs than requests")
        retry = []
        for item, output in zip(active, outputs):
            record, index = item["record"], item["record"]["inference_index"]
            if not output.finished or len(output.outputs) != 1:
                raise RuntimeError("Incomplete generation at inference_index={}".format(index))
            result = output.outputs[0]
            if not isinstance(result.text, str) or result.finish_reason not in ("stop", "length"):
                raise RuntimeError("Generation aborted at inference_index={}".format(index))
            try:
                parsed = parse_judge_response(result.text, record["metric_type"])
                error_text = None
            except (ValueError, TypeError) as error:
                parsed, error_text = None, str(error)
            item["attempts"].append({"messages": item["messages"], "response": result.text,
                                     "finish_reason": result.finish_reason, "parse_error": error_text})
            if error_text is not None and len(item["attempts"]) <= args.repair_attempts:
                item["messages"] = build_repair_messages(item["messages"], result.text, error_text)
                item["prompt"] = make_prompt(processor, item["messages"], args, index)
                retry.append(item)
                continue
            raw = {"run_fingerprint": fingerprint, "run": run, "record": record,
                   "attempts": item["attempts"], "parsed": parsed, "parse_error": error_text,
                   "physical_gpus": devices, "completed_at": datetime.now(timezone.utc).isoformat()}
            append_raw(handle, raw)
            done[index] = raw
        active = retry
    print("Judged {}/{} samples".format(len(done), total), flush=True)


def main():
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve(strict=True)
    model_path = Path(args.judge_model).expanduser().resolve(strict=True)
    samples, annotation_path = load_registered_test(args.dataset)
    records = align_predictions(read_jsonl(input_path), samples, args.dataset)
    run, fingerprint = make_run(args, input_path, annotation_path, model_path)
    raw_path, scored_path, metrics_path = prepare_artifacts(args, input_path, annotation_path)
    done = restore_raw(raw_path, records, run, fingerprint) if args.resume else {}
    pending = [row for row in records if row["inference_index"] not in done]
    if pending:
        devices = verify_gpu_selection(args)
        # Load the local processor and check every prompt before allocating the model engine.
        processor = load_processor(model_path)
        prompts = {record["inference_index"]: make_prompt(processor, build_messages(record), args,
                                                           record["inference_index"]) for record in pending}
        engine, sampling = load_engine(args, model_path)
        evaluation_error = None
        try:
            mode = "a" if args.resume else "x"
            with raw_path.open(mode, encoding="utf-8") as handle:
                if args.resume and raw_path.stat().st_size:
                    with raw_path.open("rb") as check:
                        check.seek(-1, os.SEEK_END)
                        if check.read(1) != b"\n":
                            handle.write("\n")
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    judge_batch(engine, sampling, processor, batch, prompts, args, handle, done,
                                run, fingerprint, devices, len(records))
        except BaseException as error:
            evaluation_error = error
            raise
        finally:
            try:
                shutdown_engine(engine)
            except Exception as error:
                if evaluation_error is None:
                    raise
                print("Warning: could not cleanly shut down vLLM after evaluation failed: {}"
                      .format(error), flush=True)
    scored = [scored_record(record, done[record["inference_index"]]["parsed"],
                            done[record["inference_index"]]["parse_error"]) for record in records]
    metrics = aggregate_metrics(scored)
    metrics.update({"run_fingerprint": fingerprint, "run": run,
                    "judge_parse_failures": sum(row["judge_parse_error"] is not None for row in scored),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "raw_file": str(raw_path), "scored_file": str(scored_path)})
    write_json_atomic(scored_path, scored, jsonl=True)
    write_json_atomic(metrics_path, metrics)
    print(json.dumps({key: metrics[key] for key in (
        "total", "valid", "invalid", "judge_parse_failures", "combined_flat_rawacc")}, ensure_ascii=False),
          flush=True)
    print("Saved metrics to {}".format(metrics_path), flush=True)


if __name__ == "__main__":
    main()
