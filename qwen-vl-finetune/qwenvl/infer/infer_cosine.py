"""Run vLLM offline batch inference on a registered COSINE test split.

From qwen-vl-finetune, for example:
    CUDA_VISIBLE_DEVICES=1 python -m qwenvl.infer.infer_cosine \
        --base-model /home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct \
        --dataset cosine_motion_test --output /path/to/motion.jsonl

Use --adapter-path for a language-model LoRA adapter, and --batch-size to
control the number of requests prepared and submitted together (default: 4).
Physical GPU 0 is forbidden. Selected GPU identities and PCI ordering are verified
before vLLM imports or workers start. This entry point requires vLLM >= 0.11.0.
"""

import argparse
import gc
from dataclasses import asdict
from functools import partial
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoProcessor
from transformers.utils import is_torchcodec_available
from transformers.video_utils import load_video

from qwenvl.data import data_list
from qwenvl.data.data_processor import _build_messages


ALLOWED_PHYSICAL_GPUS = {"1", "2", "3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local Qwen3-VL base model directory")
    parser.add_argument("--adapter-path", help="Optional local language-model LoRA adapter")
    parser.add_argument("--processor-path", help="Processor directory; defaults to adapter, its parent, then base")
    parser.add_argument("--dataset", required=True, help="Registered COSINE test split, e.g. cosine_motion_test")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4, help="Requests prepared and submitted per chunk")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, help="Optional vLLM context capacity; does not truncate inputs")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_runtime(args: argparse.Namespace) -> None:
    for name in ("batch_size", "tensor_parallel_size", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero")
    if args.max_model_len is not None and args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max-model-len must leave room for both the prompt and generated tokens")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be finite and greater than zero")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = [item.strip() for item in visible.split(",")]
    if (not visible or len(set(selected)) != len(selected)
            or any(item not in ALLOWED_PHYSICAL_GPUS and not item.startswith("GPU-") for item in selected)):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to unique physical GPUs 1,2,3 or their full UUIDs; physical GPU 0 is forbidden")
    if args.tensor_parallel_size != len(selected):
        raise ValueError("--tensor-parallel-size must match the number of explicitly selected GPUs")

    # nvidia-smi only queries identity; do not initialize CUDA to discover devices.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Cannot verify physical GPU identities; refusing to start vLLM") from error
    rows = [tuple(part.strip() for part in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    if any(len(row) != 3 for row in rows):
        raise RuntimeError("Cannot verify GPU PCI ordering")
    identities = {index: uuid for index, uuid, _ in rows}
    # vLLM 0.11.x requires numeric visible devices for its NVML lookup.
    # Numeric CUDA ordinals with PCI_BUS_ID must agree with physical NVML indices.
    pci_order = sorted(rows, key=lambda row: tuple(
        int(part, 16) for part in row[2].replace(".", ":").split(":")))
    if [row[0] for row in pci_order] != [str(i) for i in range(len(rows))]:
        raise RuntimeError("CUDA PCI order differs from physical GPU indices; refusing to start")
    allowed = {index: uuid for index, uuid in identities.items() if index in ALLOWED_PHYSICAL_GPUS}
    uuids = [allowed.get(item, item) for item in selected]
    if (any(uuid not in allowed.values() for uuid in uuids) or len(set(uuids)) != len(uuids)):
        raise RuntimeError("Selected GPU identity is missing, duplicated or forbidden")
    indices = [{uuid: index for index, uuid in allowed.items()}[uuid] for uuid in uuids]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"Verified physical GPUs: {dict(zip(indices, uuids))}", flush=True)


def load_cosine_samples(dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not dataset_name.startswith("cosine_") or not dataset_name.endswith("_test"):
        raise ValueError(
            "Inference accepts one registered COSINE test split only; "
            "choose cosine_whole_test or a cosine_<task>_test name."
        )

    dataset_config = data_list([dataset_name])[0]
    annotation_path = Path(dataset_config["annotation_path"])
    if annotation_path.suffix.lower() == ".jsonl":
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotations = [json.loads(line) for line in handle if line.strip()]
    else:
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)

    if not isinstance(annotations, list):
        raise TypeError("COSINE annotation file must contain a JSON list or JSONL records")

    samples = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise TypeError("Each COSINE annotation must be a JSON object")
        sample = dict(annotation)
        sample["data_path"] = dataset_config.get("data_path", "")
        samples.append(sample)
    return samples, dataset_config


def build_inference_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build messages from user turns only, preserving the shared media mapping."""
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise TypeError("Sample 'conversations' must be a list")

    user_turns = [
        turn for turn in conversations
        if isinstance(turn, dict) and turn.get("from") == "human"
    ]
    if not user_turns:
        raise ValueError("Sample has no human/user conversation turn")

    inference_sample = dict(sample)
    inference_sample["conversations"] = user_turns
    return _build_messages(inference_sample, Path(sample.get("data_path", "")))


def extract_ground_truth(sample: Dict[str, Any]) -> str:
    answers = [
        turn["value"]
        for turn in sample.get("conversations", [])
        if isinstance(turn, dict) and turn.get("from") == "gpt"
    ]
    if len(answers) != 1:
        raise ValueError(
            "Expected exactly one gpt reference answer per COSINE sample; "
            f"found {len(answers)}"
        )
    return answers[0]


def _processor_candidates(args: argparse.Namespace) -> List[str]:
    candidates = []
    if args.processor_path:
        candidates.append(args.processor_path)
    if args.adapter_path:
        adapter = Path(args.adapter_path).expanduser()
        candidates.extend([str(adapter), str(adapter.parent)])
    candidates.append(args.base_model)
    return list(dict.fromkeys(candidates))


def load_processor(args: argparse.Namespace):
    errors = []
    for candidate in _processor_candidates(args):
        try:
            processor = AutoProcessor.from_pretrained(candidate, local_files_only=True)
            if processor.__class__.__name__ != "Qwen3VLProcessor":
                raise ValueError("This entry point requires a Qwen3-VL processor")
            print(f"Loaded processor from {candidate}", flush=True)
            return processor
        except (OSError, ValueError) as error:
            if args.processor_path and candidate == args.processor_path:
                raise RuntimeError(f"Could not load explicit processor: {candidate}") from error
            errors.append(f"{candidate}: {error}")
    raise RuntimeError("Could not load processor: " + "; ".join(errors))


def preflight_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    limits = {"image": 0, "video": 0}
    for index, sample in enumerate(samples):
        try:
            extract_ground_truth(sample)
            counts = {"image": 0, "video": 0}
            for message in build_inference_messages(sample):
                for item in message["content"]:
                    modality = item["type"]
                    if modality in counts:
                        counts[modality] += 1
                        if not Path(item[modality]).is_file():
                            raise FileNotFoundError(item[modality])
            for modality in limits:
                limits[modality] = max(limits[modality], counts[modality])
        except Exception as error:
            raise ValueError(f"Invalid COSINE sample at inference_index={index}: {error}") from error
    return limits


def prepare_vllm_request(messages: List[Dict[str, Any]], processor) -> Dict[str, Any]:
    """Keep inline media order; let vLLM expand visual placeholders."""
    request = {"prompt": processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )}
    images, videos = [], []
    for message in messages:
        for item in message["content"]:
            if item["type"] == "image":
                images.append(processor.image_processor.fetch_images(item["image"]))
            elif item["type"] == "video":
                video_processor = processor.video_processor
                sample_fn = None
                if video_processor.do_sample_frames:
                    sample_fn = partial(
                        video_processor.sample_frames,
                        num_frames=video_processor.num_frames, fps=video_processor.fps,
                    )
                # Use the same decoder and sampling function as HF preprocessing.
                frames, metadata = load_video(
                    item["video"],
                    backend="torchcodec" if is_torchcodec_available() else "torchvision",
                    sample_indices_fn=sample_fn, device="cpu",
                )
                metadata = asdict(metadata)
                if hasattr(metadata["frames_indices"], "tolist"):
                    metadata["frames_indices"] = metadata["frames_indices"].tolist()
                if not metadata.get("fps") or not metadata.get("frames_indices"):
                    raise ValueError("Video lacks fps or sampled frame indices")
                metadata["do_sample_frames"] = False
                videos.append((frames, metadata))
    media = {}
    if images:
        media["image"] = images
    if videos:
        media["video"] = videos
        # Already sampled using the chosen processor; retain original timestamps.
        request["mm_processor_kwargs"] = {"do_sample_frames": False}
    if media:
        request["multi_modal_data"] = media
    return request


def prepare_model_directory(args: argparse.Namespace, processor, directory: Path) -> None:
    """Share base weights by symlink, but give vLLM the selected processor.

    vLLM loads its processor from the model directory. A temporary view keeps
    adapter/explicit image and video settings identical without touching weights.
    The caller must keep this directory alive until all requests finish.
    """
    base = Path(args.base_model).expanduser().resolve(strict=True)
    if not base.is_dir() or not (base / "config.json").is_file():
        raise ValueError("--base-model must be a local model directory containing config.json")
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Only dense Qwen3-VL models are supported by this entry point")
    processor.save_pretrained(directory)
    for source in base.iterdir():
        destination = directory / source.name
        if source.name not in {".git", ".cache"} and not destination.exists():
            destination.symlink_to(source, target_is_directory=source.is_dir())


def load_adapter_config(args: argparse.Namespace):
    if not args.adapter_path:
        return None
    adapter = Path(args.adapter_path).expanduser().resolve(strict=True)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("use_dora"):
        raise ValueError("Expected a standard LoRA adapter (DoRA is not supported here)")
    rank = config.get("r")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("Adapter must specify a positive integer LoRA rank")
    if any(value != rank for value in config.get("rank_pattern", {}).values()):
        raise ValueError("Mixed LoRA ranks are not supported by this entry point")
    targets = config.get("target_modules", [])
    if isinstance(targets, str) or not set(targets).issubset({"q_proj", "k_proj", "v_proj", "o_proj"}):
        raise ValueError("Expected the project's language-attention LoRA target modules")
    recorded_base = Path(config.get("base_model_name_or_path") or "").expanduser()
    if config.get("base_model_name_or_path") and recorded_base.exists():
        if recorded_base.resolve() != Path(args.base_model).expanduser().resolve():
            raise ValueError("Adapter base model differs from --base-model; do not mix Thinking and Instruct")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"No adapter weights found in {adapter}")
    return str(adapter), rank


def load_engine(args: argparse.Namespace, model_directory: Path, limits, adapter_config):
    try:
        from importlib.metadata import PackageNotFoundError, version
        from packaging.version import Version
        if Version(version("vllm")) < Version("0.11.0"):
            raise RuntimeError("Qwen3-VL inference requires vLLM >= 0.11.0")
        from vllm import LLM, SamplingParams
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("vLLM is not installed; configure a compatible vLLM inference environment first") from error

    kwargs = dict(
        model=str(model_directory), tokenizer=str(model_directory),
        dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size, limit_mm_per_prompt=limits, seed=args.seed,
        enable_lora=adapter_config is not None, generation_config="vllm",
    )
    if args.tensor_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "mp"
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    lora_request = None
    if adapter_config is not None:
        from vllm.lora.request import LoRARequest
        adapter_path, rank = adapter_config
        kwargs["max_lora_rank"] = max(8, 1 << (rank - 1).bit_length())
        lora_request = LoRARequest("cosine_adapter", 1, adapter_path)
    sampling = SamplingParams(
        n=1, max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0, skip_special_tokens=True,
    )
    return LLM(**kwargs), sampling, lora_request


def shutdown_vllm_engine(engine) -> None:
    """Stop vLLM workers before removing the temporary model directory."""
    llm_engine = getattr(engine, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("This vLLM version does not expose engine-core shutdown")
    shutdown()


def write_prediction(
    handle,
    dataset_name: str,
    sample_index: int,
    sample: Dict[str, Any],
    prediction: str,
    args: argparse.Namespace,
) -> None:
    meta = sample.get("_meta", {})
    output_record = {
        "dataset": dataset_name,
        "inference_index": sample_index,
        "sample_index": meta.get("sample_index", sample_index),
        "id": meta.get("id"),
        "task_type": meta.get("task_type"),
        "metric_type": meta.get("metric_type"),
        "_meta": meta,
        "ground_truth": extract_ground_truth(sample),
        "prediction": prediction,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
        },
    }
    handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    handle.flush()


def run_inference(
    engine, processor, samples: Sequence[Dict[str, Any]], dataset_name: str,
    output_path: Path, args: argparse.Namespace, sampling_params, lora_request=None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            requests = []
            for offset, sample in enumerate(batch):
                try:
                    requests.append(prepare_vllm_request(build_inference_messages(sample), processor))
                except Exception as error:
                    raise RuntimeError(f"Could not prepare inference_index={start + offset}: {error}") from error
            try:
                outputs = engine.generate(
                    requests, sampling_params=sampling_params,
                    lora_request=lora_request, use_tqdm=False,
                )
            except Exception as error:
                raise RuntimeError(
                    f"vLLM generation failed for inference_index={start}..{start + len(batch) - 1}: {error}"
                ) from error
            # LLM.generate returns one RequestOutput per input, in input order.
            if len(outputs) != len(batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} requests")
            for offset, (sample, output) in enumerate(zip(batch, outputs)):
                if not output.finished or len(output.outputs) != 1:
                    raise RuntimeError(f"Incomplete vLLM output at inference_index={start + offset}")
                write_prediction(handle, dataset_name, start + offset, sample, output.outputs[0].text.strip(), args)
            print(f"[{start + len(batch)}/{len(samples)}] inference complete", flush=True)
            del requests, outputs


def main() -> None:
    args = parse_args()
    validate_runtime(args)
    output_path = Path(args.output).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it.")
    samples, _ = load_cosine_samples(args.dataset)
    if not samples:
        raise ValueError(f"Dataset {args.dataset} contains no samples")
    limits = preflight_samples(samples)
    processor = load_processor(args)
    adapter_config = load_adapter_config(args)
    print(f"Loaded {len(samples)} samples; media limits={limits}; batch_size={args.batch_size}", flush=True)
    with TemporaryDirectory(prefix="cosine-vllm-model-") as directory:
        model_directory = Path(directory)
        prepare_model_directory(args, processor, model_directory)
        engine, sampling_params, lora_request = load_engine(args, model_directory, limits, adapter_config)
        inference_error = None
        try:
            run_inference(engine, processor, samples, args.dataset, output_path, args, sampling_params, lora_request)
        except BaseException as error:
            inference_error = error
            raise
        finally:
            try:
                shutdown_vllm_engine(engine)
            except Exception as error:
                if inference_error is None:
                    raise
                print(f"Warning: could not cleanly shut down vLLM after inference failed: {error}", flush=True)
            finally:
                engine = None
                gc.collect()
    print(f"Predictions saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
