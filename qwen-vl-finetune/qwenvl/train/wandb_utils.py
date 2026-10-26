"""Keep training GPU identities and W&B system metrics in sync."""

import csv
from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys


_ALLOWED_PHYSICAL_GPUS = {1, 2, 3}
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


@dataclass(frozen=True)
class TrainingGPUSelection:
    physical_ids: tuple[int, ...]
    uuids: tuple[str, ...]


def prepare_training_gpu_selection() -> TrainingGPUSelection:
    """Resolve physical GPU IDs before importing libraries that initialize CUDA.

    CUDA_VISIBLE_DEVICES is the only user-facing device selection. Numeric
    entries are physical NVIDIA indices; full GPU UUIDs are also accepted.
    Canonical UUIDs avoid CUDA device-order ambiguity in every child process.
    Only device identity is queried here; no CUDA context is created.
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
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + ",".join(selected),
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=15,
        )
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        if not rows or any(len(row) != 2 for row in rows):
            raise ValueError("Unexpected GPU identity response")
        by_id = {str(int(index.strip())): uuid.strip() for index, uuid in rows}
        if len(by_id) != len(rows) or any(
            not _GPU_UUID.fullmatch(uuid) for uuid in by_id.values()
        ):
            raise ValueError("Invalid or duplicated GPU identities")
        by_uuid = {uuid: int(index) for index, uuid in by_id.items()}
        physical_ids = tuple(
            int(item) if item in by_id else by_uuid[item] for item in selected
        )
        uuids = tuple(by_id[str(index)] for index in physical_ids)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        raise RuntimeError(
            "Cannot resolve training GPU identities; refusing to initialize CUDA "
            "or fall back to monitoring all GPUs."
        ) from error

    if (
        any(index not in _ALLOWED_PHYSICAL_GPUS for index in physical_ids)
        or len(set(physical_ids)) != len(physical_ids)
        or len(set(uuids)) != len(uuids)
    ):
        raise RuntimeError("Training GPU selection is duplicated or includes forbidden physical GPU 0.")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    return TrainingGPUSelection(physical_ids=physical_ids, uuids=uuids)


def configure_wandb_gpu_monitoring(training_args, selection: TrainingGPUSelection) -> None:
    """Set the native W&B GPU filter before Transformers creates its W&B run.

    W&B 0.31's NVIDIA collector uses physical NVML indices, not remapped
    logical CUDA indices. An empty filter means all GPUs, so never emit one.
    All ranks inherit the same selection; the existing WandbCallback still
    creates and owns the run on the global main process only.
    """
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [item.strip() for item in report_to.split(",")]
    if not any(str(item).lower() in {"wandb", "all"} for item in report_to):
        return

    if not selection.physical_ids or any(
        index not in _ALLOWED_PHYSICAL_GPUS for index in selection.physical_ids
    ):
        raise RuntimeError("W&B GPU monitoring requires a nonempty, allowed training GPU selection.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(selection.uuids):
        raise RuntimeError("CUDA_VISIBLE_DEVICES changed after training GPU identities were resolved.")
    wandb_module = sys.modules.get("wandb")
    if wandb_module is not None and getattr(wandb_module, "run", None) is not None:
        raise RuntimeError("Configure W&B GPU monitoring before the first wandb.init().")

    gpu_ids_json = json.dumps(list(selection.physical_ids))
    # Both supported environment spellings must agree if either was preset.
    os.environ["WANDB_X_STATS_GPU_DEVICE_IDS"] = gpu_ids_json
    os.environ["WANDB__STATS_GPU_DEVICE_IDS"] = gpu_ids_json
    if training_args.process_index == 0:
        identities = dict(zip(selection.physical_ids, selection.uuids))
        print(f"[W&B] Monitoring training physical GPUs only: {identities}", flush=True)
