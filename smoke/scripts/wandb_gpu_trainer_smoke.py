"""Short synthetic Trainer run using unmodified production helpers and WandbCallback."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("smoke_production_wandb_utils", ROOT / "qwen-vl-finetune/qwenvl/train/wandb_utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)
# Resolve the inherited UUID whitelist before importing any CUDA-aware library.
selection = utils.prepare_training_gpu_selection()

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
import wandb

class SyntheticDataset(Dataset):
    def __len__(self):
        return 4096

    def __getitem__(self, index):
        features = torch.sin(torch.arange(1024, dtype=torch.float32) + index)
        return {"features": features, "labels": features[:1]}

class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1))

    def forward(self, features, labels=None):
        time.sleep(0.35)  # Allow the real 1-second system sampler to observe training.
        output = self.layers(features)
        return {"loss": nn.functional.mse_loss(output, labels), "logits": output}

class RunAudit(TrainerCallback):
    def __init__(self):
        self.events = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.events.append({"event": "train_begin", "run_present": wandb.run is not None, "world_zero": state.is_world_process_zero})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.events.append({"event": "log", "step": state.global_step, "logs": logs})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.relative_to((ROOT / "smoke/results").resolve())
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert tuple(selection.physical_ids) == tuple(json.loads(os.environ["SMOKE_EXPECTED_PHYSICAL_GPUS"]))
    training_args = TrainingArguments(
        output_dir=str(result_dir / "trainer"), report_to=["wandb"],
        run_name="wandb-gpu-monitoring-smoke", max_steps=24,
        per_device_train_batch_size=16, gradient_accumulation_steps=1,
        learning_rate=1e-4, logging_steps=1, save_strategy="no",
        eval_strategy="no", disable_tqdm=True, dataloader_num_workers=0,
        dataloader_pin_memory=False, optim="adamw_torch", seed=42,
        ddp_find_unused_parameters=False, local_rank=local_rank,
    )
    utils.configure_wandb_gpu_monitoring(training_args, selection)
    assert wandb.run is None, "W&B must be initialized by Transformers WandbCallback"
    assert torch.cuda.device_count() == len(selection.physical_ids)
    device = training_args.device
    props = torch.cuda.get_device_properties(device)
    actual_uuid = str(props.uuid)
    expected_uuid = selection.uuids[device.index]
    assert actual_uuid.lower().removeprefix("gpu-") == expected_uuid.lower().removeprefix("gpu-"), (actual_uuid, expected_uuid)
    identity = {
        "rank": rank, "local_rank": local_rank, "pid": os.getpid(),
        "physical_gpu": selection.physical_ids[device.index], "cuda_device": str(device),
        "uuid": actual_uuid, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "physical_filter": list(selection.physical_ids), "device_count": torch.cuda.device_count(),
    }
    (result_dir / f"rank-{rank}-identity.json").write_text(json.dumps(identity, indent=2))
    audit = RunAudit()
    torch.manual_seed(42)
    model = TinyRegressor()
    initial = next(model.parameters()).detach().clone()
    trainer = Trainer(model=model, args=training_args, train_dataset=SyntheticDataset(), callbacks=[audit])
    callback_names = [type(x).__name__ for x in trainer.callback_handler.callbacks]
    assert "WandbCallback" in callback_names
    output = trainer.train()
    changed = not torch.equal(initial, next(model.parameters()).detach().cpu())
    record = {"rank": rank, "global_step": output.global_step, "training_loss": output.training_loss,
              "weights_changed": changed, "callbacks": callback_names, "audit": audit.events,
              "run_present": wandb.run is not None}
    if rank == 0:
        run = wandb.run
        assert run is not None
        assert list(run.settings.x_stats_gpu_device_ids) == list(selection.physical_ids)
        metrics = {}
        # Read the native collector's actual buffer; do not synthesize GPU metrics.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            metrics = run._system_metrics
            if callable(metrics):
                metrics = metrics()
            selected = {int(m.group(1)) for key in metrics if (m := re.search(r"(?:^|[.])gpu[.]([0-9]+)[.]", key))}
            gpu_samples = [len(samples) for key, samples in metrics.items() if re.search(r"(?:^|[.])gpu[.]([0-9]+)[.]", key)]
            if selected and gpu_samples and min(gpu_samples) >= 3:
                break
            time.sleep(1)
        serializable = {key: [[stamp.isoformat(), value] for stamp, value in values] for key, values in metrics.items()}
        (result_dir / "system-metrics.json").write_text(json.dumps(serializable, indent=2))
        record.update({"run_id": run.id, "mode": run.settings.mode, "wandb_dir": run.dir,
                       "settings_gpu_ids": list(run.settings.x_stats_gpu_device_ids),
                       "stats_sampling_interval": run.settings.x_stats_sampling_interval,
                       "summary": dict(run.summary)})
        run.finish()
    assert output.global_step == 24 and math.isfinite(output.training_loss) and changed
    (result_dir / f"rank-{rank}-result.json").write_text(json.dumps(record, indent=2))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
