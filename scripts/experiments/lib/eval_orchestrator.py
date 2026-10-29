"""Run a frozen experiment's complete eval plan through one Qwen judge job."""

from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

from .artifacts import check_manifest, make_manifest
from .common import TASKS, fingerprint, read_json, require_id, sha256_file, utc_now, write_json_atomic
from .config import resolve_job_config
from .eval_executor import _inputs, _make_request, _planned_job, build_eval_command
from .eval_worker import inspect_evaluation, pair_id
from .infer_orchestrator import SUPPORTED_SETTINGS, _artifact_map, _mark, _signals
from .orchestration_logging import run_logged
from .records import load_experiment, load_job, load_stage_run, record_status, update_status

if __package__ == "lib":
    from pipeline import build_stage_plan
else:
    from ..pipeline import build_stage_plan


def _resources(config):
    resources = []
    for index, group in enumerate(config["eval"]["run"]["gpu_groups"]):
        if (not group or len(group) != config["eval"]["params"]["tensor_parallel_size"]
                or any(type(gpu) is not int or gpu not in (1, 2, 3) for gpu in group)
                or len(set(group)) != len(group)):
            raise ValueError("Eval groups must use unique physical GPUs 1-3 matching tensor_parallel_size")
        resources.append({"group_index": index, "physical_gpu_ids": list(group)})
    if not resources:
        raise ValueError("Eval requires at least one configured GPU group")
    return resources


def _context(run_dir):
    run, snapshot = load_stage_run(run_dir)
    config = snapshot["config"]
    settings = config["experiment"]
    experiment_dir = Path(settings["records_root"]).resolve() / run["experiment_id"]
    experiment, experiment_snapshot = load_experiment(experiment_dir)
    registration = {"stage": "eval", "run_id": run["run_id"],
                    "path": run_dir.relative_to(experiment_dir).as_posix()}
    if (run["stage"] != "eval" or settings["setting"] not in SUPPORTED_SETTINGS
            or not {"infer", "eval"}.issubset(settings["enabled_stages"])
            or run_dir != experiment_dir / "stage-runs" / "eval" / run["run_id"]
            or experiment["experiment_id"] != run["experiment_id"]
            or experiment["status"] != "active" or run["status"] not in ("planned", "running")
            or settings != experiment_snapshot["config"]["experiment"]
            or snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or [item for item in experiment["stage_runs"] if item["stage"] == "eval"] != [registration]
            or run["plan"] != "plan.json"
            or Path(run["output_root"]) != Path(settings["eval_root"]) / run["experiment_id"] / run["run_id"]):
        raise ValueError("Eval run is not an active, registered child of its frozen experiment")
    plan = read_json(run_dir / "plan.json")
    if (plan != build_stage_plan(snapshot, "eval")
            or fingerprint(plan) != run["plan_fingerprint"] or len(plan["jobs"]) != 1
            or plan["jobs"][0]["job_id"] != "eval-judge-all"
            or run["jobs"] != [{"job_id": "eval-judge-all", "path": "jobs/eval-judge-all"}]):
        raise ValueError("Eval requires its complete frozen plan and exactly one judge job")
    job_dir = run_dir / "jobs" / "eval-judge-all"
    job, job_snapshot = load_job(job_dir)
    resolved = resolve_job_config(snapshot, "eval", job["job_id"],
                                  job_snapshot["layers"].get("job"), job_snapshot["layers"].get("job_cli"))
    _planned_job(job, plan["jobs"][0])
    if (job["job_id"] != "eval-judge-all" or job["stage"] != "eval"
            or job["run_id"] != run["run_id"] or job["experiment_id"] != run["experiment_id"]
            or job["config_fingerprint"] != resolved["config_fingerprint"]
            or job_snapshot["layers"].get("parent_snapshot") != snapshot["snapshot_fingerprint"]
            or job_snapshot["config"]["experiment"] != settings
            or job["training_state_policy"] is not None or job["resume_training_state"]
            or job["tasks"] != list(TASKS) or job["task"] is not None or job["depends_on"]
            or job["lineage"].get("judge_model_loads") != 1
            or job["output_spec"] != {"artifact_kind": "metrics", "slot": "combined"}
            or Path(job["output_dir"]) != Path(run["output_root"]) / job["job_id"]
            or job["status"] not in ("planned", "succeeded")):
        raise ValueError("Eval job differs from its frozen plan, effective config or allowed lifecycle")
    if job["status"] == "planned":
        evidence = [job_dir / "execution.json", job_dir / "evaluation.request.json", Path(job["output_dir"])]
        if job["inputs"] or job["artifacts"] or any(path.exists() or path.is_symlink() for path in evidence):
            raise ValueError("Planned eval job already has inputs or execution evidence; review it manually")
    info = {"directory": job_dir, "record": job, "snapshot": job_snapshot,
            "initial_status": job["status"], "resources": _resources(job_snapshot["config"])}
    _bind_predictions(info, experiment_dir, experiment, experiment_snapshot)
    return run, snapshot, experiment_dir, info


def _bind_predictions(info, experiment_dir, experiment, experiment_snapshot):
    registrations = [item for item in experiment["stage_runs"] if item["stage"] == "infer"]
    if len(registrations) != 1:
        raise ValueError("Eval requires this experiment's unique successful infer run")
    registration = registrations[0]
    infer_dir = experiment_dir / "stage-runs" / "infer" / require_id(registration["run_id"])
    source_run, source_snapshot = load_stage_run(infer_dir)
    settings = info["snapshot"]["config"]["experiment"]
    if (registration["path"] != infer_dir.relative_to(experiment_dir).as_posix()
            or source_run["stage"] != "infer" or source_run["status"] != "succeeded"
            or source_run["experiment_id"] != experiment["experiment_id"]
            or source_run["run_id"] != registration["run_id"]
            or source_snapshot["config"]["experiment"] != settings
            or source_snapshot["layers"].get("parent_snapshot") != experiment_snapshot["snapshot_fingerprint"]
            or source_run["plan"] != "plan.json"
            or Path(source_run["output_root"]) != Path(settings["infer_root"]) / experiment["experiment_id"] / source_run["run_id"]):
        raise ValueError("Registered infer source is incomplete or belongs to another frozen experiment")
    plan = read_json(infer_dir / "plan.json")
    if (plan != build_stage_plan(source_snapshot, "infer")
            or source_run["plan_fingerprint"] != fingerprint(plan)
            or source_run["jobs"] != [{"job_id": item["job_id"], "path": "jobs/" + item["job_id"]}
                                      for item in plan["jobs"]]):
        raise ValueError("Successful infer registration differs from its complete frozen plan")
    expected_pairs = [{"prediction_job": item["job_id"], "prediction_slot": item["output"]["slot"], "task": task}
                      for item in plan["jobs"] for task in item["tasks"]]
    pairs = info["record"]["lineage"]["scoring_pairs"]
    if pairs != expected_pairs:
        raise ValueError("Eval scoring pairs do not cover the exact successful infer plan in order")
    sources, predictions = {}, {}
    for pair in pairs:
        identifier = pair["prediction_job"]
        if identifier not in sources:
            directory = infer_dir / "jobs" / identifier
            job, _ = load_job(directory)
            if Path(job["output_dir"]) != Path(source_run["output_root"]) / identifier:
                raise ValueError("Infer source output is outside its registered run")
            sources[identifier] = {"directory": directory, "record": job,
                                   "artifacts": _artifact_map(directory, job)}
        source = sources[identifier]
        role = require_id("prediction-" + pair_id(pair))
        artifact_role = "prediction-" + pair["task"]
        manifest = source["artifacts"].get(artifact_role)
        if manifest is None or manifest["kind"] != "prediction":
            raise ValueError("Missing registered prediction for " + pair_id(pair))
        predictions[role] = {"manifest": manifest,
                             "manifest_path": str(source["directory"] / "artifacts" / (artifact_role + ".json"))}
    info.update(infer_run_id=source_run["run_id"], sources=sources, predictions=predictions)
    _resolved_inputs(info, experiment_dir)


def _resolved_inputs(info, experiment_dir):
    """Reuse bound-input inspection without binding or mutating the job record."""
    job = info["record"]
    if job["status"] == "succeeded":
        inputs = deepcopy(job["inputs"])
        actual = [value for value in inputs.values() if value["kind"] == "prediction"]
        expected = [value["manifest"] for value in info["predictions"].values()]
        if (len(actual) != len(expected)
                or sorted(fingerprint(value) for value in actual) != sorted(fingerprint(value) for value in expected)):
            raise ValueError("Successful eval job used predictions different from the registered infer results")
    else:
        inputs = {role: deepcopy(value["manifest"]) for role, value in info["predictions"].items()}
        for spec in job["input_specs"]:
            inputs[spec["role"]] = make_manifest(spec["kind"], spec["path"],
                                                 source=spec["source"], metadata=spec["metadata"])
    if not inputs:
        raise ValueError("Eval requires explicitly resolved inputs")
    # _inputs inspects an already-bound copy. Its bind_inputs branch is never
    # reached: no prospective inputs are written by the high-level process.
    candidate = {**job, "inputs": inputs}
    _, inputs, pairs = _inputs(info["directory"], candidate, info["snapshot"], {}, experiment_dir)
    return inputs, pairs


def _eval_artifacts(info):
    job, directory = info["record"], info["directory"]
    result = {}
    producer = {key: job[key] for key in ("experiment_id", "stage", "run_id", "job_id")}
    for item in job["artifacts"]:
        role = require_id(item["role"])
        if role in result or item["path"] != "artifacts/" + role + ".json":
            raise ValueError("Duplicate or redirected eval artifact registration")
        manifest = check_manifest(read_json(directory / item["path"]), verify_contents=True)
        if (manifest["kind"] != item["kind"] or manifest["fingerprint"] != item["fingerprint"]
                or manifest["producer"] != producer):
            raise ValueError("Eval artifact differs from its registered producer")
        result[role] = manifest
    return result


def _audit_succeeded_job(info, experiment_dir):
    job, snapshot = load_job(info["directory"])
    if (job["status"] != "succeeded"
            or snapshot["snapshot_fingerprint"] != info["snapshot"]["snapshot_fingerprint"]
            or any(job[key] != info["record"][key] for key in
                   ("experiment_id", "stage", "run_id", "job_id", "config_fingerprint",
                    "task", "tasks", "depends_on", "requires", "input_specs", "lineage", "output_spec", "output_dir",
                    "training_state_policy", "resume_training_state"))):
        raise ValueError("Successful eval job identity or frozen configuration changed")
    info["record"] = job
    inputs, pairs = _resolved_inputs(info, experiment_dir)
    config = snapshot["config"]
    execution = read_json(info["directory"] / "execution.json")
    index = execution.get("gpu_group_index")
    if (execution.get("schema_version") != 1 or execution.get("status") != "succeeded"
            or execution.get("cleanup_completed") is not True or execution.get("cleanup_error")
            or execution.get("returncode") != 0 or type(index) is not int
            or not 0 <= index < len(info["resources"])
            or execution.get("physical_gpu_ids") != info["resources"][index]["physical_gpu_ids"]
            or execution.get("job_id") != job["job_id"]
            or execution.get("config_fingerprint") != job["config_fingerprint"]
            or execution.get("tasks") != job["tasks"]
            or execution.get("scoring_pairs") != job["lineage"]["scoring_pairs"]
            or execution.get("output_dir") != job["output_dir"]):
        raise ValueError("Successful eval execution or worker cleanup is unconfirmed")
    output = Path(job["output_dir"])
    request_path = info["directory"] / "evaluation.request.json"
    request = read_json(request_path)
    expected = _make_request(job, config, inputs, pairs, output, execution["physical_gpu_ids"], execution["gpu_identity"])
    if (request != expected or read_json(output / "evaluation.request.json") != request
            or read_json(output / "execution.json") != execution
            or execution.get("request_fingerprint") != request["request_fingerprint"]
            or execution.get("input_fingerprints") != request["input_fingerprints"]
            or execution.get("judge_model_fingerprint") != request["judge_model_fingerprint"]
            or execution.get("command") != build_eval_command(config, request_path)
            or execution.get("cwd") != config["experiment"]["repo_root"]):
        raise ValueError("Successful eval request differs from its exact frozen inputs or receipt")
    summary = inspect_evaluation(request)
    outputs = _eval_artifacts(info)
    common = {"config_fingerprint": job["config_fingerprint"], "lineage": job["lineage"],
              "request_fingerprint": request["request_fingerprint"], "input_fingerprints": request["input_fingerprints"],
              "params": request["params"], "judge_model_fingerprint": request["judge_model_fingerprint"],
              "judge_processor_fingerprint": request["judge_processor_fingerprint"],
              "prompt_version": summary["prompt_version"], "judge_model_loads": 1}
    expected_outputs = {}
    for pair, result in zip(pairs, summary["pairs"]):
        metadata = {**common, **{key: pair[key] for key in
                    ("pair_id", "prediction_job", "prediction_slot", "task", "dataset", "prediction_fingerprint",
                     "prediction_producer", "annotation_fingerprint")}, "sample_count": result["total"]}
        for name, kind, prefix in (("judge.raw.jsonl", "judge_raw", "judge-raw-"),
                                   ("judge.scored.jsonl", "judge_scored", "judge-scored-"),
                                   ("judge.metrics.json", "summary", "pair-metrics-")):
            expected_outputs[prefix + pair["pair_id"]] = (kind, Path(pair["output_dir"]) / name, metadata)
    metadata = {**common, "slot": "combined", "tasks": job["tasks"], "pair_order": summary["pair_order"],
                "sample_count": summary["total"], "combined_flat_rawacc": summary["combined_flat_rawacc"]}
    expected_outputs.update({"metrics": ("metrics", output / "judge.metrics.json", metadata),
                             "evaluation-summary": ("summary", output / "evaluation_result.json", metadata)})
    if (set(outputs) != set(expected_outputs)
            or execution.get("artifacts") != {role: value["fingerprint"] for role, value in outputs.items()}):
        raise ValueError("Successful eval is missing planned pair or combined artifacts")
    for role, (kind, path, metadata) in expected_outputs.items():
        manifest = outputs[role]
        if (manifest["kind"] != kind or manifest["metadata"] != metadata
                or manifest["requested_path"] != str(path) or manifest["path"] != str(path.resolve(strict=True))):
            raise ValueError("Successful eval artifact path or metadata differs: " + role)
    info.update(result=summary, outputs=outputs, request=request, execution=execution,
                returncode=execution["returncode"], assignment=info["resources"][index])
    return summary


@contextmanager
def _run_lock(run_dir):
    with (run_dir / ".eval-orchestrator.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another high-level eval run is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _launch(info, invocation_dir, receipt):
    config = info["snapshot"]["config"]
    repo_root = Path(config["experiment"]["repo_root"])
    low_level = repo_root / "scripts/experiments/run_eval_job.py"
    if not low_level.is_file():
        raise ValueError("Low-level eval entry point is missing: " + str(low_level))
    # There is exactly one judge job. Use the first frozen group; the low-level
    # executor waits for its shared device locks instead of querying GPU load.
    resource = info["resources"][0]
    command = [config["eval"]["run"]["python"], str(low_level),
               "--job-dir", str(info["directory"]), "--gpu-group-index", str(resource["group_index"])]
    for role, source in info["predictions"].items():
        command.extend(["--input", role + "=" + source["manifest_path"]])
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, resource["physical_gpu_ids"])),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    log_path = invocation_dir / "eval-judge-all.log"
    launch = {"job_id": info["record"]["job_id"], "command": command,
              "cwd": str(repo_root), "assignment": resource, "log": str(log_path),
              "started_at": utc_now(), "pid": None, "status": "launching"}
    receipt["launches"].append(launch)
    info.update(assignment=resource, log_path=str(log_path))
    write_json_atomic(invocation_dir / "execution.json", receipt)
    log = log_path.open("xb")
    try:
        print("Starting eval-judge-all on physical GPUs {}; log: {}".format(
            resource["physical_gpu_ids"], log_path), flush=True)
        process = subprocess.Popen(command, cwd=str(repo_root), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException as error:
        log.close()
        launch.update(status="launch_failed", error=str(error), finished_at=utc_now())
        raise
    # Retain the child before any further operation that can raise, so the
    # enclosing finally always requests low-level worker cleanup and waits.
    info.update(process=process, log_handle=log)
    launch.update(pid=process.pid, status="running")
    write_json_atomic(invocation_dir / "execution.json", receipt)


def _finish_child(info, receipt, *, stop=False):
    process = info.get("process")
    if process is None:
        return
    try:
        if stop and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        # The low-level executor owns its worker process group. Do not kill
        # that executor while it is shutting down vLLM or writing evidence.
        info["returncode"] = process.wait()
    finally:
        info["log_handle"].close()
    receipt["launches"][-1].update(returncode=info["returncode"], finished_at=utc_now(),
                                     status=record_status(info["directory"])["status"])


def _check_exit(info):
    record, snapshot = load_job(info["directory"])
    execution_path = info["directory"] / "execution.json"
    execution = read_json(execution_path) if execution_path.is_file() else {}
    if (info["returncode"] != 0 or record["status"] != "succeeded"
            or snapshot["snapshot_fingerprint"] != info["snapshot"]["snapshot_fingerprint"]
            or execution.get("cleanup_completed") is not True or execution.get("cleanup_error")):
        raise RuntimeError("Low-level eval exited {} with status {}; cleanup_completed={}; cleanup_error={}".format(
            info["returncode"], record["status"], execution.get("cleanup_completed"), execution.get("cleanup_error")))


def _result_rows(info, experiment_config):
    setting = experiment_config["setting"]
    order = experiment_config["task_order"]
    rows = []
    for result in info["result"]["pairs"]:
        source = info["sources"][result["prediction_job"]]
        job = source["record"]
        lineage = job["lineage"]
        adapter = job["inputs"].get("adapter")
        adapter_path = None
        if adapter is not None:
            producer = adapter["producer"]
            adapter_path = str(Path(experiment_config["records_root"]) / producer["experiment_id"] /
                               "stage-runs/tune" / producer["run_id"] / "jobs" /
                               producer["job_id"] / "artifacts/adapter.json")
        slot, task = result["prediction_slot"], result["task"]
        if setting in ("transfer_round1", "transfer_round2"):
            row_key, column_key = lineage["source_task"], lineage["second_round_task"]
        else:
            row_key, column_key = slot, task
        metric_role = "pair-metrics-" + result["pair_id"]
        prediction = source["artifacts"]["prediction-" + task]
        annotation = info["record"]["inputs"]["annotation-{}-test".format(task)]
        rows.append({**deepcopy(result), "setting": setting,
                     "matrix_row": row_key, "matrix_column": column_key,
                     "train_task": slot if setting in ("cl", "singletask", "multitask")
                     else lineage.get("second_round_task"),
                     "cl_stage": order.index(slot) + 1 if setting == "cl" else None,
                     "source_task": lineage.get("source_task"),
                     "second_round_task": lineage.get("second_round_task"),
                     "test_task": task, "adapter_ref": lineage.get("adapter_ref"),
                     "infer_lineage": deepcopy(lineage),
                     "adapter": deepcopy(adapter),
                     "adapter_manifest": adapter_path,
                     "prediction_producer": prediction["producer"],
                     "prediction_manifest": str(source["directory"] / "artifacts" / ("prediction-" + task + ".json")),
                     "prediction_path": prediction["path"], "annotation_path": annotation["path"],
                     "metrics_manifest": str(info["directory"] / "artifacts" / (metric_role + ".json")),
                     "metrics_fingerprint": info["outputs"][metric_role]["fingerprint"]})
    if ([row["pair_id"] for row in rows] != info["result"]["pair_order"]
            or len(rows) != len(info["record"]["lineage"]["scoring_pairs"])):
        raise ValueError("Summary rows do not match the complete frozen scoring pair order")
    return rows


def _matrix(rows, experiment_config):
    setting = experiment_config["setting"]
    row_order = (list(experiment_config["task_order"]) if setting == "cl" else
                 ["base"] if setting == "zero_shot" else ["whole"] if setting == "multitask" else list(TASKS))
    columns = list(TASKS)
    values = [[None for _ in columns] for _ in row_order]
    counts = [[None for _ in columns] for _ in row_order]
    pair_ids = [[None for _ in columns] for _ in row_order]
    for row in rows:
        ri, ci = row_order.index(row["matrix_row"]), columns.index(row["matrix_column"])
        if pair_ids[ri][ci] is not None:
            raise ValueError("Multiple eval pairs map to the same result matrix cell")
        value = row["combined_flat_rawacc"]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Invalid combined_flat_rawacc in result matrix")
        values[ri][ci], counts[ri][ci], pair_ids[ri][ci] = value, row["total"], row["pair_id"]
    expected = {(ri, ci) for ri in range(len(row_order)) for ci in range(len(columns))
                if setting in ("zero_shot", "multitask", "cl")
                or (setting == "singletask" and ri == ci)
                or (setting in ("transfer_round1", "transfer_round2") and ri != ci)}
    populated = {(ri, ci) for ri, line in enumerate(pair_ids) for ci, value in enumerate(line) if value is not None}
    if populated != expected:
        raise ValueError("Result matrix does not cover every planned setting cell")
    return {"metric": "combined_flat_rawacc", "unit": "fraction", "row_order": row_order,
            "column_order": columns,
            "row_axis": "source_task" if setting.startswith("transfer_") else "adapter_slot",
            "column_axis": "second_round_task" if setting.startswith("transfer_") else "test_task",
            "test_task_rule": "source_task" if setting == "transfer_round1" else
                              "second_round_task" if setting == "transfer_round2" else "column_task",
            "values": values, "sample_counts": counts, "pair_ids": pair_ids,
            "unmeasured_cells": "null in JSON, empty in CSV; never a zero score"}


def _write_text_atomic(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix="." + path.name + "-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_results(info, invocation_dir, experiment_config):
    rows = _result_rows(info, experiment_config)
    matrix = _matrix(rows, experiment_config)
    evaluated = info["result"]
    keys = ("total", "valid", "invalid", "score_sum", "combined_flat_rawacc", "judge_parse_failures")
    result = {"schema_version": 1, "kind": "experiment_eval_results", "status": "succeeded",
              "producer": evaluated["producer"], "setting": experiment_config["setting"],
              "task_order": experiment_config["task_order"],
              "config_fingerprint": info["record"]["config_fingerprint"],
              "request_fingerprint": evaluated["request_fingerprint"],
              "judge_model_fingerprint": evaluated["judge_model_fingerprint"],
              "input_fingerprints": evaluated["input_fingerprints"],
              "model_loads": evaluated["model_loads"],
              "pair_order": evaluated["pair_order"], "pair_count": len(rows),
              "aggregation_scope": evaluated["aggregation_scope"],
              "aggregation_rule": "score_sum / total over all prediction/sample pairs; not the mean of task accuracies",
              "overall": {key: evaluated[key] for key in keys}, "rows": rows, "matrix": matrix,
              "metrics_manifest": str(info["directory"] / "artifacts/metrics.json"),
              "metrics_fingerprint": info["outputs"]["metrics"]["fingerprint"],
              "metrics_file": evaluated["metrics_file"], "metrics_sha256": evaluated["metrics_sha256"]}
    write_json_atomic(invocation_dir / "matrix.json", matrix)
    fields = ["pair_id", "setting", "prediction_job", "prediction_slot", "cl_stage", "train_task",
              "source_task", "second_round_task", "test_task", "matrix_row", "matrix_column", *keys,
              "adapter_ref", "adapter_manifest", "prediction_manifest", "prediction_path",
              "annotation_path", "metrics_manifest", "prediction_fingerprint", "annotation_fingerprint",
              "metrics_fingerprint", "output_dir"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _write_text_atomic(invocation_dir / "results.csv", buffer.getvalue())
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([matrix["row_axis"] + " / " + matrix["column_axis"], *matrix["column_order"]])
    writer.writerows([[label, *values] for label, values in zip(matrix["row_order"], matrix["values"])])
    _write_text_atomic(invocation_dir / "matrix.csv", buffer.getvalue())
    write_json_atomic(invocation_dir / "results.json", result)
    paths = {name: {"path": str(invocation_dir / name), "sha256": sha256_file(invocation_dir / name)}
             for name in ("results.json", "results.csv", "matrix.json", "matrix.csv")}
    return result, paths


def _summary(run_dir, invocation_dir, run, snapshot, info, started_at, status, details, results, files):
    record = record_status(info["directory"])
    result = {"schema_version": 1, "kind": "eval_run_summary",
              "experiment_id": run["experiment_id"], "run_id": run["run_id"],
              "setting": snapshot["config"]["experiment"]["setting"], "status": status,
              "config_fingerprint": run["config_fingerprint"], "infer_run_id": info["infer_run_id"],
              "invocation_dir": str(invocation_dir), "started_at": started_at, "finished_at": utc_now(),
              "job_order": [record["job_id"]], "scoring_pairs": record["lineage"]["scoring_pairs"],
              "job": {"job_id": record["job_id"], "status": record["status"],
                      "initial_status": info["initial_status"], "output_dir": record["output_dir"],
                      "config_fingerprint": record["config_fingerprint"],
                      "reused_successful_job": info["initial_status"] == "succeeded",
                      "returncode": info.get("returncode"), "log": info.get("log_path"),
                      "assignment": info.get("assignment"),
                      "artifacts": [{**artifact, "manifest": str(info["directory"] / artifact["path"])}
                                    for artifact in record["artifacts"]]},
              "details": details, "result_files": files if status == "succeeded" else {}}
    if status == "succeeded":
        result.update(overall=results["overall"], pair_count=results["pair_count"],
                      combined_flat_rawacc=results["overall"]["combined_flat_rawacc"],
                      invalid=results["overall"]["invalid"],
                      model_loads=results["model_loads"], matrix=results["matrix"])
    write_json_atomic(invocation_dir / "summary.json", result)
    write_json_atomic(run_dir / "eval-summary.json", result)
    return result


def execute_eval_run(run_dir):
    """Score one complete frozen experiment with a single low-level judge load."""
    return run_logged("eval", run_dir, _execute_eval_run)


def _execute_eval_run(run_dir, invocation_dir):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    with _run_lock(run_dir), _signals() as signals:
        run, snapshot, experiment_dir, info = _context(run_dir)
        started_at = utc_now()
        config = snapshot["config"]
        receipt = {"schema_version": 1, "experiment_id": run["experiment_id"], "run_id": run["run_id"],
                   "config_fingerprint": run["config_fingerprint"], "started_at": started_at,
                   "pid": os.getpid(), "status": "running", "launches": [],
                   "max_parallel_jobs": config["eval"]["run"]["max_parallel_jobs"],
                   "fail_fast": config["eval"]["run"]["fail_fast"],
                   "job_order": [info["record"]["job_id"]], "infer_run_id": info["infer_run_id"],
                   "input_manifests": {role: item["manifest_path"] for role, item in info["predictions"].items()}}
        write_json_atomic(invocation_dir / "execution.json", receipt)
        status, details, results, files = "failed", [], None, {}
        try:
            if run["status"] == "planned":
                update_status(run_dir, "running", "High-level experiment eval started")
            if signals["number"] is not None:
                raise InterruptedError("Eval orchestrator interrupted during preparation")
            if info["initial_status"] == "planned":
                _launch(info, invocation_dir, receipt)
                while info["process"].poll() is None:
                    if signals["number"] is not None:
                        raise InterruptedError("Eval orchestrator received signal {}".format(signals["number"]))
                    time.sleep(0.1)
                _finish_child(info, receipt)
                write_json_atomic(invocation_dir / "execution.json", receipt)
                _check_exit(info)
            if signals["number"] is not None:
                raise InterruptedError("Eval orchestrator interrupted before result inspection")
            _audit_succeeded_job(info, experiment_dir)
            if signals["number"] is not None:
                raise InterruptedError("Eval orchestrator interrupted during result inspection")
            results, files = _publish_results(info, invocation_dir, config["experiment"])
            if signals["number"] is not None:
                raise InterruptedError("Eval orchestrator interrupted during result publication")
            status = "succeeded"
        except (KeyboardInterrupt, InterruptedError) as error:
            status = "cancelled"
            details.append({"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()})
        except Exception as error:
            details.append({"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()})
        finally:
            try:
                _finish_child(info, receipt, stop=True)
            except Exception as error:
                status = "failed"
                details.append({"type": type(error).__name__,
                                "message": "Child cleanup failed: " + str(error),
                                "traceback": traceback.format_exc()})
        if status != "succeeded":
            # Only discard this invocation's derived exports. Preserve every
            # low-level raw judgment, metric, log and execution receipt.
            for name in ("results.json", "results.csv", "matrix.json", "matrix.csv"):
                (invocation_dir / name).unlink(missing_ok=True)
            _mark(info["directory"], status, "High-level eval {}: {}".format(status, details))
        summary = _summary(run_dir, invocation_dir, run, snapshot, info, started_at, status, details, results, files)
        update_status(run_dir, status, "All planned eval pairs inspected and summarized" if status == "succeeded"
                      else "High-level eval run " + status)
        receipt.update(status=status, finished_at=utc_now(), summary=str(invocation_dir / "summary.json"))
        write_json_atomic(invocation_dir / "execution.json", receipt)
        experiment, _ = load_experiment(experiment_dir)
        if (status == "succeeded" and experiment["status"] == "active" and experiment["stage_runs"]
                and all(record_status(experiment_dir / item["path"])["status"] == "succeeded"
                        for item in experiment["stage_runs"])):
            update_status(experiment_dir, "succeeded", "All registered stage runs completed")
        return summary
