"""Forward Trainer's W&B logs to SwanLab and sample only the training GPUs."""

import os
from pathlib import Path
import warnings

from transformers.integrations import WandbCallback

from qwenvl.train.wandb_utils import TrainingGPUSelection, configure_wandb_gpu_monitoring


class SwanLabWandbCallback(WandbCallback):
    """Keep the Trainer integration; add selected-card metrics before forwarding."""

    def __init__(self, selection: TrainingGPUSelection, *, monitor_enabled=True):
        super().__init__()
        self.selection = selection
        self.monitor_enabled = monitor_enabled
        self._warned = set()

    def _warn_once(self, key, error):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(f"SwanLab GPU metric {key} unavailable: {error}", RuntimeWarning)

    def _gpu_metrics(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(self.selection.uuids):
            raise RuntimeError("Training GPU visibility changed before monitoring.")
        import pynvml

        metrics = {}
        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            for index, gpu_uuid in zip(self.selection.physical_ids, self.selection.uuids):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
                    actual_index = int(pynvml.nvmlDeviceGetIndex(handle))
                except pynvml.NVMLError as error:
                    self._warn_once(f"gpu/{index}/identity", error)
                    continue
                if actual_index != index or actual_index not in {1, 2, 3}:
                    raise RuntimeError("Monitoring GPU identity differs from the allowed training selection.")
                prefix = f"gpu/{index}/"
                try:
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics[prefix + "memory_used_mib"] = memory.used / (1024 ** 2)
                    metrics[prefix + "memory_total_mib"] = memory.total / (1024 ** 2)
                except pynvml.NVMLError as error:
                    self._warn_once(prefix + "memory", error)
                readers = {
                    "utilization_pct": lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    "power_w": lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "temperature_c": lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                }
                for name, read in readers.items():
                    try:
                        metrics[prefix + name] = read()
                    except pynvml.NVMLError as error:
                        self._warn_once(prefix + name, error)
        except pynvml.NVMLError as error:
            self._warn_once("nvml", error)
        finally:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except pynvml.NVMLError as error:
                    self._warn_once("nvml_shutdown", error)
        return metrics

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        combined = dict(logs or {})
        if state.is_world_process_zero and self.monitor_enabled:
            combined.update(self._gpu_metrics())
        return super().on_log(args, state, control, model=model, logs=combined, **kwargs)

    def finish(self):
        """Flush the synchronized runs after final checkpoints have been saved."""
        if self._wandb.run is not None:
            self._wandb.finish()


def configure_swanlab_monitoring(training_args, selection: TrainingGPUSelection):
    """Install the official sync hook before the Trainer's first wandb.init().

    Legacy YAML report_to/project/mode fields remain compatible. Only global
    rank zero initializes SwanLab; native hardware probes in both SDKs are
    disabled to avoid enumerating physical GPU 0 or unrelated devices.
    """
    report_to = training_args.report_to or []
    if isinstance(report_to, str):
        report_to = [item.strip() for item in report_to.split(",")]
    if not any(str(item).lower() in {"wandb", "all"} for item in report_to):
        return None
    configure_wandb_gpu_monitoring(training_args, selection)
    mode = os.environ.get("SWANLAB_MODE", os.environ.get("WANDB_MODE", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("Monitoring mode must be online, offline or disabled.")
    os.environ["SWANLAB_MODE"] = mode
    os.environ["WANDB_MODE"] = "offline"
    for key in ("WANDB_X_DISABLE_STATS", "WANDB__DISABLE_STATS",
                "WANDB_X_DISABLE_META", "WANDB_X_DISABLE_MACHINE_INFO"):
        os.environ[key] = "true"

    if training_args.process_index == 0:
        import swanlab

        if swanlab.has_run():
            raise RuntimeError("Configure SwanLab synchronization before creating any run.")
        log_dir = Path(os.environ.get("SWANLAB_LOG_DIR", str(Path(training_args.output_dir) / "logs" / "swanlab")))
        swanlab.merge_settings(swanlab.Settings(
            interactive=False,
            probe=swanlab.Settings.Probe(hardware=False, monitor=False),
            # Fresh SwanLab ID for each job, including an explicit checkpoint restore.
            run=swanlab.Settings.Run(id=None, resume="never"),
            experiment=swanlab.Settings.Experiment(group=os.environ.get("WANDB_RUN_GROUP")),
        ))
        swanlab.sync_wandb(mode=mode, wandb_run=False, log_dir=str(log_dir))
        print(f"[SwanLab] mode={mode}; W&B compatibility is offline; logs={log_dir}", flush=True)
    return SwanLabWandbCallback(selection, monitor_enabled=mode != "disabled")
