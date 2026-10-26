"""Launch offline smoke training with an explicit physical GPU whitelist.

Example: python -B smoke/scripts/run_wandb_gpu_monitoring.py --devices 2,3 --result-dir smoke/results/RUN/ddp-gpu23
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True, help="Physical GPU IDs 1-3 only")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate saved evidence without launching training or querying GPUs")
    args = parser.parse_args()
    selected = args.devices.split(",")
    if not selected or len(set(selected)) != len(selected) or any(x not in {"1", "2", "3"} for x in selected):
        parser.error("Use a nonempty unique subset of physical GPUs 1,2,3; GPU 0 is forbidden")
    result_dir = args.result_dir.resolve()
    result_dir.relative_to((ROOT / "smoke/results").resolve())
    if args.check_only:
        launch = json.loads((result_dir / "launch.json").read_text())
        if launch["physical_ids"] != [int(x) for x in selected]:
            parser.error("Saved launch does not match requested physical GPUs")
        previous = json.loads((result_dir / "verdict.json").read_text())
        selection = SimpleNamespace(physical_ids=tuple(launch["physical_ids"]), uuids=tuple(launch["uuids"]))
        check_results(result_dir, selection, previous["launcher_exit_code"], previous["elapsed_seconds"])
        return
    result_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    spec = importlib.util.spec_from_file_location("smoke_production_wandb_utils", ROOT / "qwen-vl-finetune/qwenvl/train/wandb_utils.py")
    utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = utils
    spec.loader.exec_module(utils)
    selection = utils.prepare_training_gpu_selection()
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(selection.uuids),
        "SMOKE_EXPECTED_PHYSICAL_GPUS": json.dumps(list(selection.physical_ids)),
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2", "WANDB_MODE": "offline",
        "WANDB_PROJECT": "qwen3vl-wandb-gpu-smoke", "WANDB_LOG_MODEL": "false",
        "WANDB_WATCH": "false", "WANDB_DISABLE_GIT": "true", "WANDB_DISABLE_CODE": "true",
        "WANDB_CONSOLE": "off", "WANDB_X_STATS_SAMPLING_INTERVAL": "1",
        "WANDB__STATS_SAMPLING_INTERVAL": "1", "WANDB_X_STATS_BUFFER_SIZE": "1024",
        "WANDB__STATS_BUFFER_SIZE": "1024", "HF_HUB_OFFLINE": "1",
    })
    for key, folder in {
        "WANDB_DIR": "wandb", "WANDB_CACHE_DIR": "wandb-cache", "WANDB_CONFIG_DIR": "wandb-config",
        "WANDB_DATA_DIR": "wandb-data", "HF_HOME": "hf-cache", "XDG_CACHE_HOME": "cache",
    }.items():
        target = result_dir / folder
        target.mkdir()
        env[key] = str(target)
    # Unix sockets have a short pathname limit. Keep TMPDIR relative and short,
    # while retaining all temporary artifacts below the ignored results tree.
    short_tmp = Path(tempfile.mkdtemp(prefix=".wg-", dir=ROOT / "smoke/results"))
    env["TMPDIR"] = str(short_tmp.relative_to(ROOT))
    (result_dir / "runtime-tmp").symlink_to(short_tmp, target_is_directory=True)
    worker = ROOT / "smoke/scripts/wandb_gpu_trainer_smoke.py"
    command = [sys.executable, "-B", str(worker), "--result-dir", str(result_dir)]
    if len(selected) > 1:
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={len(selected)}", str(worker), "--result-dir", str(result_dir)]
    launch = {"command": command, "physical_ids": list(selection.physical_ids), "uuids": list(selection.uuids),
              "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "WANDB_MODE", "WANDB_X_STATS_SAMPLING_INTERVAL", "WANDB_X_STATS_BUFFER_SIZE", "TMPDIR"]}}
    (result_dir / "launch.json").write_text(json.dumps(launch, indent=2))
    start = time.monotonic()
    with (result_dir / "launcher.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    check_results(result_dir, selection, exit_code, time.monotonic() - start)

def check_results(result_dir, selection, exit_code, elapsed):
    failures = []
    if exit_code:
        failures.append(f"Trainer launcher exited {exit_code}")
    rank_results = []
    for rank in range(len(selection.physical_ids)):
        path = result_dir / f"rank-{rank}-result.json"
        identity_path = result_dir / f"rank-{rank}-identity.json"
        if not path.exists() or not identity_path.exists():
            failures.append(f"rank {rank}: missing result/identity evidence")
            continue
        record = json.loads(path.read_text())
        identity = json.loads(identity_path.read_text())
        rank_results.append(record)
        if record["run_present"] != (rank == 0):
            failures.append(f"rank {rank}: W&B run ownership mismatch")
        begin = next((event for event in record["audit"] if event["event"] == "train_begin"), {})
        if begin.get("run_present") != (rank == 0):
            failures.append(f"rank {rank}: incorrect W&B ownership at train begin")
        if rank == 0:
            summary = record.get("summary", {})
            if summary.get("train/global_step") != 24 or not math.isfinite(summary.get("train/loss", math.nan)):
                failures.append("rank 0: missing actual W&B training metrics")
            if record.get("settings_gpu_ids") != list(selection.physical_ids) or record.get("mode") != "offline":
                failures.append("rank 0: unexpected W&B GPU filter or run mode")
        if record["global_step"] != 24 or not record["weights_changed"]:
            failures.append(f"rank {rank}: training not completed")
        if identity["physical_gpu"] != selection.physical_ids[rank] or identity["CUDA_VISIBLE_DEVICES"] != ",".join(selection.uuids):
            failures.append(f"rank {rank}: unexpected GPU identity or visibility")
    if len(list((result_dir / "wandb").glob("wandb/offline-run-*"))) != 1:
        failures.append("Expected exactly one offline W&B run")
    metrics_path = result_dir / "system-metrics.json"
    summaries = {}
    metric_ids = set()
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for key, values in metrics.items():
            match = re.search(r"(?:^|[.])gpu[.](process[.])?([0-9]+)[.](.+)", key)
            if match:
                gpu, metric = int(match.group(2)), (match.group(1) or "") + match.group(3)
                metric_ids.add(gpu)
                numbers = [float(value) for _, value in values]
                summaries.setdefault(str(gpu), {})[metric] = {"samples": len(values), "min": min(numbers), "max": max(numbers)}
        if metric_ids != set(selection.physical_ids):
            failures.append(f"collected GPU IDs {sorted(metric_ids)} != selected physical IDs {list(selection.physical_ids)}")
        for gpu in selection.physical_ids:
            values = summaries.get(str(gpu), {})
            for metric in ["gpu", "memoryAllocated", "powerWatts", "temp"]:
                if metric not in values or values[metric]["samples"] < 3:
                    failures.append(f"physical GPU {gpu}: missing >=3 real samples for {metric}")
            for metric in ["memoryAllocated", "powerWatts", "temp"]:
                if metric in values and values[metric]["max"] <= 0:
                    failures.append(f"physical GPU {gpu}: {metric} has no positive measurement")
    else:
        failures.append("missing native system metrics")
    verdict = {"passed": not failures, "launcher_exit_code": exit_code, "elapsed_seconds": elapsed,
               "expected_physical_ids": list(selection.physical_ids), "observed_metric_ids": sorted(metric_ids),
               "gpu_metrics": summaries, "failures": failures}
    (result_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    raise SystemExit(0 if verdict["passed"] else 1)

if __name__ == "__main__":
    main()
