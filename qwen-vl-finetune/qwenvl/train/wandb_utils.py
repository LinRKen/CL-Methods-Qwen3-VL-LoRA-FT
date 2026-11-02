"""Resolve training GPU identities and preserve the W&B compatibility filter."""

from dataclasses import dataclass
import json
import os
import re
import sys


_ALLOWED_PHYSICAL_GPUS = {1, 2, 3}
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


@dataclass(frozen=True)
class TrainingGPUSelection:
    physical_ids: tuple[int, ...]
    uuids: tuple[str, ...]


def prepare_training_gpu_selection() -> TrainingGPUSelection:
    """Resolve only selected GPU identities through NVML, before CUDA imports.

    CUDA_VISIBLE_DEVICES accepts physical indices 1-3 or full GPU UUIDs.
    Never enumerate the machine or fall back to another device. NVML queries
    device identity without creating a CUDA context and avoids the nvidia-smi
    device-selection path on hosts with unrelated device errors.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = [item.strip() for item in visible.split(",")]
    if not visible.strip() or any(not item for item in selected):
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES explicitly to physical GPUs 1, 2, 3 or "
            "their full UUIDs before launching training; physical GPU 0 is forbidden."
        )
    if len(set(selected)) != len(selected) or any(
        item not in {"1", "2", "3"} and not _GPU_UUID.fullmatch(item)
        for item in selected
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain unique physical GPU IDs 1, 2, 3 "
            "or full GPU UUIDs; physical GPU 0 is forbidden."
        )

    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError("Training GPU identity checks require nvidia-ml-py.") from error

    physical_ids, uuids = [], []
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        for item in selected:
            handle = (pynvml.nvmlDeviceGetHandleByIndex(int(item)) if item.isdecimal()
                      else pynvml.nvmlDeviceGetHandleByUUID(item))
            index = int(pynvml.nvmlDeviceGetIndex(handle))
            if index not in _ALLOWED_PHYSICAL_GPUS:
                raise ValueError("Forbidden physical GPU identity: {}".format(index))
            value = pynvml.nvmlDeviceGetUUID(handle)
            gpu_uuid = value.decode("ascii") if isinstance(value, bytes) else value
            if not isinstance(gpu_uuid, str) or not _GPU_UUID.fullmatch(gpu_uuid):
                raise ValueError("Invalid GPU UUID response")
            if (item.isdecimal() and index != int(item)) or (
                not item.isdecimal() and gpu_uuid.lower() != item.lower()
            ):
                raise ValueError("NVML identity disagrees with the requested GPU")
            physical_ids.append(index)
            uuids.append(gpu_uuid)
    except (pynvml.NVMLError, OSError, ValueError) as error:
        raise RuntimeError(
            "Cannot resolve allowed training GPU identities through NVML; "
            "refusing to initialize CUDA or fall back to other GPUs."
        ) from error
    finally:
        if initialized:
            pynvml.nvmlShutdown()

    if len(set(physical_ids)) != len(physical_ids) or len(set(uuids)) != len(uuids):
        raise RuntimeError("Training GPU selection contains duplicate physical devices.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    return TrainingGPUSelection(physical_ids=tuple(physical_ids), uuids=tuple(uuids))


def configure_wandb_gpu_monitoring(training_args, selection: TrainingGPUSelection) -> None:
    """Validate the common selection before the W&B/SwanLab compatibility hook.

    Keep W&B's physical NVML index filter for existing callers. The SwanLab
    adapter disables native system probes and samples this same selection.
    """
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [item.strip() for item in report_to.split(",")]
    if not any(str(item).lower() in {"wandb", "all"} for item in report_to):
        return

    if (not selection.physical_ids
            or len(selection.physical_ids) != len(selection.uuids)
            or len(set(selection.physical_ids)) != len(selection.physical_ids)
            or len(set(selection.uuids)) != len(selection.uuids)
            or any(index not in _ALLOWED_PHYSICAL_GPUS for index in selection.physical_ids)
            or any(not _GPU_UUID.fullmatch(value) for value in selection.uuids)):
        raise RuntimeError("Monitoring requires a nonempty, unique, allowed training GPU selection.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(selection.uuids):
        raise RuntimeError("CUDA_VISIBLE_DEVICES changed after training GPU identities were resolved.")
    wandb_module = sys.modules.get("wandb")
    if wandb_module is not None and getattr(wandb_module, "run", None) is not None:
        raise RuntimeError("Configure training monitoring before the first wandb.init().")

    gpu_ids_json = json.dumps(list(selection.physical_ids))
    os.environ["WANDB_X_STATS_GPU_DEVICE_IDS"] = gpu_ids_json
    os.environ["WANDB__STATS_GPU_DEVICE_IDS"] = gpu_ids_json
    if training_args.process_index == 0:
        identities = dict(zip(selection.physical_ids, selection.uuids))
        print(f"[Monitor] Training physical GPU selection: {identities}", flush=True)
