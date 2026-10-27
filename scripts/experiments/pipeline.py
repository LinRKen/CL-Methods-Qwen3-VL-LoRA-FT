"""Plan and register COSINE experiment runs without executing model workloads.

The setting-specific plans capture run hierarchy, task relations and artifact lineage.
Execution backends for tune, infer and eval are deliberately outside this module.
"""

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    from lib.common import SCHEMA_VERSION, STAGES, TASKS, require_id, write_json_atomic
    from lib.config import (
        ConfigError, dotted_overrides, load_config, resolve_job_config, validate_snapshot,
    )
    from lib.records import create_experiment, create_job, create_stage_run
else:
    from .lib.common import SCHEMA_VERSION, STAGES, TASKS, require_id, write_json_atomic
    from .lib.config import (
        ConfigError, dotted_overrides, load_config, resolve_job_config, validate_snapshot,
    )
    from .lib.records import create_experiment, create_job, create_stage_run

WHOLE_TASK = "whole"


def _artifact_ref(stage, job_id, artifact_kind, *, slot=None, task=None):
    reference = {"stage": stage, "job_id": require_id(job_id), "artifact_kind": artifact_kind}
    if slot is not None:
        reference["slot"] = require_id(slot)
    if task is not None:
        reference["task"] = require_id(task)
    return reference


def _job(job_id, *, task=None, tasks=None, depends_on=None, requires=None,
         input_specs=None, lineage=None, output=None):
    result = {
        "job_id": require_id(job_id),
        "depends_on": list(depends_on or []),
        "requires": list(requires or []),
        "input_specs": list(input_specs or []),
        "lineage": {} if lineage is None else lineage,
    }
    if task is not None:
        result["task"] = task
    if tasks is not None:
        result["tasks"] = list(tasks)
    if output is not None:
        result["output"] = dict(output)
        if output["artifact_kind"] == "prediction":
            result["output"]["tasks"] = list(tasks or [])
    return result


def _annotation_spec(experiment, task, split):
    from pathlib import Path
    return {
        "role": "annotation-{}-{}".format(task, split),
        "kind": "annotation",
        "path": str(Path(experiment["dataset_root"]) / task / (split + ".json")),
        "source": "configuration:experiment.dataset_root",
        "metadata": {"task": task, "split": split},
    }


def _model_specs(experiment, include_processor=True):
    specs = [{
        "role": "base-model", "kind": "base_model", "path": experiment["base_model"],
        "source": "configuration:experiment.base_model", "metadata": {},
    }]
    if include_processor:
        specs.append({
            "role": "processor", "kind": "processor", "path": experiment["processor_path"],
            "source": "configuration:experiment.processor_path", "metadata": {},
        })
    return specs


def _tune_input_specs(experiment, task):
    return _model_specs(experiment) + [_annotation_spec(experiment, task, "train")]


def _infer_input_specs(experiment, tasks):
    return _model_specs(experiment) + [
        _annotation_spec(experiment, task, "test") for task in tasks
    ]


def _eval_input_specs(experiment):
    return [
        {"role": "judge-model", "kind": "judge_model", "path": experiment["judge_model"],
         "source": "configuration:experiment.judge_model", "metadata": {}},
        {"role": "judge-processor", "kind": "processor", "path": experiment["judge_model"],
         "source": "configuration:experiment.judge_model", "metadata": {}},
        *[_annotation_spec(experiment, task, "test") for task in TASKS],
    ]


def _tune_jobs(config):
    experiment = config["experiment"]
    setting = experiment["setting"]
    order = list(experiment["task_order"])
    jobs = []
    if setting == "cl":
        previous = None
        for index, task in enumerate(order, 1):
            job_id = "tune-{:02d}-{}".format(index, task)
            jobs.append(_job(
                job_id,
                task=task,
                input_specs=_tune_input_specs(experiment, task),
                depends_on=[] if previous is None else [previous],
                requires=[] if previous is None else [
                    _artifact_ref("tune", previous, "adapter", slot=order[index - 2]),
                ],
                lineage={
                    "train_task": task,
                    "initialization": "base_model" if previous is None else "previous_adapter_weights",
                    "parent_job": previous,
                    "training_state": "fresh_optimizer_scheduler_epoch_global_step",
                },
                output={"artifact_kind": "adapter", "slot": task},
            ))
            previous = job_id
        return jobs
    if setting == "multitask":
        return [_job("tune-whole", task=WHOLE_TASK,
                     input_specs=_tune_input_specs(experiment, WHOLE_TASK), lineage={
            "train_task": WHOLE_TASK,
            "initialization": "base_model",
            "training_state": "fresh_optimizer_scheduler_epoch_global_step",
        }, output={"artifact_kind": "adapter", "slot": WHOLE_TASK})]
    if setting == "singletask":
        return [_job("tune-{}".format(task), task=task, lineage={
            "train_task": task,
            "initialization": "base_model",
            "training_state": "fresh_optimizer_scheduler_epoch_global_step",
        }, output={"artifact_kind": "adapter", "slot": task},
            input_specs=_tune_input_specs(experiment, task)) for task in TASKS]
    if setting in ("transfer_round1", "transfer_round2"):
        for task in TASKS:
            job_id = "tune-round1-{}".format(task)
            jobs.append(_job(job_id, task=task, input_specs=_tune_input_specs(experiment, task), lineage={
                "round": 1,
                "train_task": task,
                "initialization": "base_model",
                "training_state": "fresh_optimizer_scheduler_epoch_global_step",
            }, output={"artifact_kind": "adapter", "slot": "round1-{}".format(task)}))
        for source in TASKS:
            parent = "tune-round1-{}".format(source)
            for target in TASKS:
                if target == source:
                    continue
                job_id = "tune-round2-{}-to-{}".format(source, target)
                jobs.append(_job(job_id, task=target, input_specs=_tune_input_specs(experiment, target),
                    depends_on=[parent], requires=[
                    _artifact_ref("tune", parent, "adapter", slot="round1-{}".format(source)),
                ], lineage={
                    "round": 2,
                    "source_task": source,
                    "train_task": target,
                    "initialization": "round1_adapter_weights",
                    "parent_job": parent,
                    "training_state": "fresh_optimizer_scheduler_epoch_global_step",
                }, output={"artifact_kind": "adapter", "slot": "round2-{}-to-{}".format(source, target)}))
        return jobs
    if setting == "zero_shot":
        return []
    raise ConfigError("Unsupported setting: {}".format(setting))


def _infer_jobs(config):
    experiment = config["experiment"]
    setting = experiment["setting"]
    if setting == "zero_shot":
        return [_job("infer-base", tasks=TASKS, input_specs=_infer_input_specs(experiment, TASKS), lineage={
            "model": experiment["base_model"],
            "adapter": None,
            "test_tasks": list(TASKS),
            "model_loads_per_job": 1,
        }, output={"artifact_kind": "prediction", "slot": "base"})]
    if setting == "cl":
        jobs = []
        for item in _tune_jobs(config):
            task = item["task"]
            slot = item["output"]["slot"]
            jobs.append(_job(
                "infer-{}".format(task),
                tasks=TASKS,
                input_specs=_infer_input_specs(experiment, TASKS),
                requires=[_artifact_ref("tune", item["job_id"], "adapter", slot=slot)],
                lineage={"adapter_ref": item["job_id"], "test_tasks": list(TASKS), "model_loads_per_job": 1},
                output={"artifact_kind": "prediction", "slot": slot},
            ))
        return jobs
    if setting == "multitask":
        return [_job("infer-whole", tasks=TASKS, input_specs=_infer_input_specs(experiment, TASKS), requires=[
            _artifact_ref("tune", "tune-whole", "adapter", slot=WHOLE_TASK),
        ], lineage={
            "adapter_ref": "tune-whole", "test_tasks": list(TASKS), "model_loads_per_job": 1,
        }, output={"artifact_kind": "prediction", "slot": WHOLE_TASK})]
    if setting == "singletask":
        return [_job("infer-{}".format(task), tasks=[task],
                     input_specs=_infer_input_specs(experiment, [task]), requires=[
            _artifact_ref("tune", "tune-{}".format(task), "adapter", slot=task),
        ], lineage={
            "adapter_ref": "tune-{}".format(task), "test_tasks": [task], "model_loads_per_job": 1,
        }, output={"artifact_kind": "prediction", "slot": task}) for task in TASKS]
    if setting in ("transfer_round1", "transfer_round2"):
        jobs = []
        for source in TASKS:
            for target in TASKS:
                if target == source:
                    continue
                adapter_job = "tune-round2-{}-to-{}".format(source, target)
                test_task = source if setting == "transfer_round1" else target
                slot = "round2-{}-to-{}".format(source, target)
                jobs.append(_job(
                    "infer-{}".format(slot),
                    tasks=[test_task],
                    input_specs=_infer_input_specs(experiment, [test_task]),
                    requires=[_artifact_ref("tune", adapter_job, "adapter", slot=slot)],
                    lineage={
                        "adapter_ref": adapter_job,
                        "source_task": source,
                        "second_round_task": target,
                        "test_tasks": [test_task],
                        "model_loads_per_job": 1,
                    },
                    output={"artifact_kind": "prediction", "slot": slot},
                ))
        return jobs
    raise ConfigError("Unsupported setting: {}".format(setting))


def _eval_jobs(config):
    inference_jobs = _infer_jobs(config)
    scoring_pairs = []
    prediction_refs = []
    for item in inference_jobs:
        output_slot = item["output"]["slot"]
        for task in item.get("tasks", []):
            prediction_refs.append(_artifact_ref("infer", item["job_id"], "prediction", slot=output_slot, task=task))
            scoring_pairs.append({
                "prediction_job": item["job_id"],
                "prediction_slot": output_slot,
                "task": task,
            })
    return [_job(
        "eval-judge-all",
        tasks=TASKS,
        input_specs=_eval_input_specs(config["experiment"]),
        requires=prediction_refs,
        lineage={
            "judge_model": config["experiment"]["judge_model"],
            "scoring_pairs": scoring_pairs,
            "judge_model_loads": 1,
        },
        output={"artifact_kind": "metrics", "slot": "combined"},
    )]


def build_stage_plan(snapshot, stage):
    validate_snapshot(snapshot)
    if stage not in STAGES:
        raise ConfigError("Unknown stage: {}".format(stage))
    experiment = snapshot["config"]["experiment"]
    if stage not in experiment["enabled_stages"]:
        raise ConfigError("Stage is not enabled: {}".format(stage))
    jobs = {"tune": _tune_jobs, "infer": _infer_jobs, "eval": _eval_jobs}[stage](snapshot["config"])
    configured_jobs = set(snapshot["config"][stage]["jobs"])
    planned_jobs = {item["job_id"] for item in jobs}
    unused = configured_jobs - planned_jobs
    if unused:
        raise ConfigError("Overrides refer to jobs absent from this plan: {}".format(sorted(unused)))
    # Resolve all overrides before any experiment/run/job records are created.
    for item in jobs:
        resolve_job_config(snapshot, stage, item["job_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "setting": experiment["setting"],
        "config_fingerprint": snapshot["config_fingerprint"],
        "high_level_policy": {
            "run_count": 1,
            "low_level_job_count": len(jobs),
            "execution": "deferred",
        },
        "jobs": jobs,
    }


def build_plan(snapshot):
    validate_snapshot(snapshot)
    experiment = snapshot["config"]["experiment"]
    stages = {stage: build_stage_plan(snapshot, stage) for stage in experiment["enabled_stages"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment["name"],
        "setting": experiment["setting"],
        "task_order": list(experiment["task_order"]),
        "config_fingerprint": snapshot["config_fingerprint"],
        "stages": stages,
    }


def prepare_records(snapshot, experiment_id=None):
    """Materialize a configuration and its plans as records; launch no workloads."""
    validate_snapshot(snapshot)
    plan = build_plan(snapshot)
    experiment_dir = create_experiment(snapshot, experiment_id=experiment_id)
    write_json_atomic(experiment_dir / "plan.json", plan)
    for stage, stage_plan in plan["stages"].items():
        run_dir = create_stage_run(experiment_dir, stage, stage_plan=stage_plan)
        for item in stage_plan["jobs"]:
            create_job(
                run_dir,
                item["job_id"],
                task=item.get("task"),
                tasks=item.get("tasks"),
                depends_on=item.get("depends_on", []),
                requires=item.get("requires", []),
                input_specs=item.get("input_specs", []),
                lineage=item.get("lineage", {}),
                output=item.get("output"),
            )
    return experiment_dir


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True, type=Path, help="setting YAML")
        sub.add_argument("--default", dest="default_path", type=Path, default=None,
                         help="default YAML; omitted uses configs/default.yaml")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="dotted YAML override")
        if command == "prepare":
            sub.add_argument("--experiment-id", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    overrides = dotted_overrides(args.set) if args.set else {}
    snapshot = load_config(args.config, default_path=args.default_path, cli_overrides=overrides)
    if args.command == "plan":
        print(json.dumps(build_plan(snapshot), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    experiment_dir = prepare_records(snapshot, experiment_id=args.experiment_id)
    print(str(experiment_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

