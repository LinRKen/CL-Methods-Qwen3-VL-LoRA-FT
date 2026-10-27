"""Strict YAML loading, inheritance and immutable effective-config snapshots."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import yaml

from .common import SCHEMA_VERSION, STAGES, TASKS, fingerprint, require_id

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
SETTINGS = ("cl", "zero_shot", "multitask", "singletask", "transfer_round1", "transfer_round2")


class ConfigError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    """Reject repeated mapping keys instead of silently discarding a parameter."""


def _construct_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError("Duplicate YAML key: {}".format(key))
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def deep_merge(base, override):
    """Merge mappings recursively; every other value, including a list, replaces."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# Leaf names are deliberate: no arbitrary kwargs, environment variables or shell commands.
EXPERIMENT_FIELDS = {
    "name": "id", "setting": "setting", "tasks": "tasks", "task_order": "task_order",
    "enabled_stages": "stages", "repo_root": "path", "dataset_root": "path",
    "base_model": "path", "processor_path": "path", "judge_model": "path",
    "records_root": "path", "checkpoints_root": "path", "infer_root": "path", "eval_root": "path",
}
TUNE_FIELDS = {
    "num_train_epochs": "positive", "per_device_train_batch_size": "positive_int",
    "gradient_accumulation_steps": "positive_int", "learning_rate": "positive",
    "weight_decay": "nonnegative", "warmup_ratio": "ratio", "max_grad_norm": "positive",
    "lr_scheduler_type": ("cosine", "linear", "constant", "constant_with_warmup"),
    "optim": ("adamw_torch", "adamw_torch_fused"), "bf16": "bool", "fp16": "bool",
    "model_max_length": "positive_int", "gradient_checkpointing": "bool",
    "dataloader_num_workers": "nonnegative_int", "logging_steps": "positive_int",
    "seed": "nonnegative_int", "data_seed": "nonnegative_int", "data_flatten": "bool",
    "data_packing": "bool", "tune_mm_vision": "bool", "tune_mm_mlp": "bool",
    "tune_mm_llm": "bool", "lora_enable": "bool", "lora_r": "positive_int",
    "lora_alpha": "positive_int", "lora_dropout": "dropout", "min_pixels": "positive_int",
    "max_pixels": "positive_int", "video_min_frames": "positive_int", "video_max_frames": "positive_int",
    "video_min_pixels": "positive_int", "video_max_pixels": "positive_int", "video_fps": "positive",
    "base_interval": "positive_int", "eval_strategy": ("no",), "save_strategy": ("epoch", "steps"),
    "save_steps": "positive_int", "save_total_limit": "positive_int",
    "report_to": ("wandb", "none"), "deepspeed": "text",
}
INFER_FIELDS = {
    "batch_size": "positive_int", "dtype": ("bfloat16", "float16", "float32"),
    "tensor_parallel_size": "positive_int", "gpu_memory_utilization": "positive_ratio",
    "max_model_len": "positive_int", "max_new_tokens": "positive_int", "do_sample": "bool",
    "temperature": "positive", "top_p": "positive_ratio", "seed": "nonnegative_int",
    "min_pixels": "positive_int", "max_pixels": "positive_int", "video_min_frames": "positive_int",
    "video_max_frames": "positive_int", "video_min_pixels": "positive_int",
    "video_max_pixels": "positive_int", "video_fps": "positive",
}
EVAL_FIELDS = {
    "batch_size": "positive_int", "dtype": ("bfloat16", "float16", "float32"),
    "tensor_parallel_size": "positive_int", "gpu_memory_utilization": "positive_ratio",
    "max_model_len": "positive_int", "max_new_tokens": "positive_int",
    "repair_attempts": "nonnegative_int", "seed": "nonnegative_int",
}
COMMON_RUN_FIELDS = {
    "python": "path", "gpu_groups": "gpu_groups", "max_parallel_jobs": "positive_int", "fail_fast": "bool",
}
TUNE_RUN_FIELDS = {
    **COMMON_RUN_FIELDS, "master_addr": "text", "master_port": "port",
    "wandb_project": "text", "wandb_mode": ("online", "offline", "disabled"),
}
PARAM_FIELDS = {"tune": TUNE_FIELDS, "infer": INFER_FIELDS, "eval": EVAL_FIELDS}
RUN_FIELDS = {"tune": TUNE_RUN_FIELDS, "infer": COMMON_RUN_FIELDS, "eval": COMMON_RUN_FIELDS}


def _fail(key, message):
    raise ConfigError("{}: {}".format(key, message))


def _check_leaf(value, rule, key):
    if value is None:
        _fail(key, "null is not supported; omit the key to inherit its default")
    if isinstance(rule, tuple):
        if type(value) is not str or value not in rule:
            _fail(key, "expected one of {}".format(rule))
    elif rule == "bool":
        if type(value) is not bool:
            _fail(key, "expected a YAML boolean")
    elif rule in ("positive_int", "nonnegative_int", "port"):
        minimum = 0 if rule == "nonnegative_int" else 1
        if type(value) is not int or value < minimum or (rule == "port" and value > 65535):
            _fail(key, "invalid integer for {}".format(rule))
    elif rule in ("positive", "nonnegative", "ratio", "positive_ratio", "dropout"):
        if type(value) not in (int, float) or not math.isfinite(value):
            _fail(key, "expected a finite number")
        if rule in ("positive", "positive_ratio") and value <= 0:
            _fail(key, "must be positive")
        if rule in ("nonnegative", "ratio", "dropout") and value < 0:
            _fail(key, "cannot be negative")
        if rule in ("ratio", "positive_ratio") and value > 1:
            _fail(key, "cannot exceed 1")
        if rule == "dropout" and value >= 1:
            _fail(key, "must be less than 1")
    elif rule in ("text", "path", "id", "setting"):
        if type(value) is not str or not value.strip() or value != value.strip():
            _fail(key, "expected non-empty text without surrounding whitespace")
        if rule == "path" and (not Path(value).is_absolute() or ".." in Path(value).parts):
            _fail(key, "expected an absolute path without '..'; no environment expansion is performed")
        if rule == "id":
            require_id(value)
        if rule == "setting" and value not in SETTINGS:
            _fail(key, "unsupported setting")
    elif rule in ("tasks", "task_order", "stages"):
        choices = STAGES if rule == "stages" else TASKS
        if (not isinstance(value, list) or not value or any(type(item) is not str or item not in choices for item in value)
                or len(set(value)) != len(value)):
            _fail(key, "expected a non-empty list of distinct values from {}".format(choices))
        if rule in ("tasks", "task_order") and set(value) != set(TASKS):
            _fail(key, "must contain all six COSINE tasks exactly once")
    elif rule == "gpu_groups":
        if not isinstance(value, list) or not value:
            _fail(key, "expected groups such as [[2, 3]] or [[1], [2]]")
        seen = set()
        for group in value:
            if (not isinstance(group, list) or not group
                    or any(type(gpu) is not int or gpu not in (1, 2, 3) for gpu in group)
                    or len(set(group)) != len(group) or seen.intersection(group)):
                _fail(key, "groups must be non-empty and disjoint physical GPU subsets of 1,2,3; GPU 0 is forbidden")
            seen.update(group)
    else:
        raise AssertionError("Unknown field rule: {}".format(rule))


def _check_mapping(value, fields, key, partial=False):
    if not isinstance(value, dict):
        _fail(key, "expected a mapping")
    unknown = set(value) - set(fields)
    missing = set(fields) - set(value)
    if unknown:
        _fail(key, "unknown keys: {}".format(sorted(unknown)))
    if missing and not partial:
        _fail(key, "missing keys: {}".format(sorted(missing)))
    for name, item in value.items():
        _check_leaf(item, fields[name], "{}.{}".format(key, name))


def validate_job_override(value, stage):
    if stage not in STAGES:
        raise ConfigError("Unknown stage: {}".format(stage))
    if not isinstance(value, dict) or set(value) - {"params", "run"}:
        raise ConfigError("Job overrides accept only params and run")
    for section, fields in (("params", PARAM_FIELDS[stage]), ("run", RUN_FIELDS[stage])):
        if section in value:
            _check_mapping(value[section], fields, "{}.jobs.{}".format(stage, section), partial=True)


def validate_config(value, partial=False):
    if not isinstance(value, dict):
        raise ConfigError("Config must be a mapping")
    root = {"experiment", *STAGES}
    if set(value) - root or (not partial and set(value) != root):
        raise ConfigError("Root keys must be experiment, tune, infer, eval")
    if "experiment" in value:
        _check_mapping(value["experiment"], EXPERIMENT_FIELDS, "experiment", partial)
    for stage in STAGES:
        if stage not in value:
            continue
        section = value[stage]
        if not isinstance(section, dict) or set(section) - {"params", "run", "jobs"}:
            _fail(stage, "expected params, run and jobs mappings")
        if not partial and set(section) != {"params", "run", "jobs"}:
            _fail(stage, "params, run and jobs must all be present after inheritance")
        for name, fields in (("params", PARAM_FIELDS[stage]), ("run", RUN_FIELDS[stage])):
            if name in section:
                _check_mapping(section[name], fields, stage + "." + name, partial)
        if "jobs" in section:
            if not isinstance(section["jobs"], dict):
                _fail(stage + ".jobs", "expected a mapping keyed by low-level job ID")
            for job_id, override in section["jobs"].items():
                require_id(job_id)
                validate_job_override(override, stage)
    if partial:
        return
    experiment = value["experiment"]
    if experiment["setting"] == "zero_shot" and "tune" in experiment["enabled_stages"]:
        _fail("experiment.enabled_stages", "zero_shot cannot enable tune")
    for stage in STAGES:
        params, run = value[stage]["params"], value[stage]["run"]
        if run["max_parallel_jobs"] > len(run["gpu_groups"]):
            _fail(stage + ".run.max_parallel_jobs", "cannot exceed the number of disjoint GPU groups")
        if stage != "tune":
            if any(len(group) != params["tensor_parallel_size"] for group in run["gpu_groups"]):
                _fail(stage + ".run.gpu_groups", "each group size must equal tensor_parallel_size")
            if params["max_model_len"] <= params["max_new_tokens"]:
                _fail(stage + ".params.max_model_len", "must exceed max_new_tokens")
        for lower, upper in (("min_pixels", "max_pixels"), ("video_min_frames", "video_max_frames"),
                             ("video_min_pixels", "video_max_pixels")):
            if lower in params and params[lower] > params[upper]:
                _fail(stage + ".params." + lower, "cannot exceed " + upper)
    tune = value["tune"]["params"]
    if tune["bf16"] and tune["fp16"]:
        _fail("tune.params", "bf16 and fp16 cannot both be true")
    if not tune["lora_enable"] or tune["tune_mm_vision"]:
        _fail("tune.params", "this pipeline requires LoRA with a frozen vision encoder")
    if tune["data_flatten"] and tune["data_packing"]:
        _fail("tune.params", "data_flatten and data_packing cannot both be true")


def _load_yaml(path):
    path = Path(path).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)
    except (yaml.YAMLError, UnicodeError) as error:
        raise ConfigError("{}: {}".format(path, error)) from error
    validate_config(value, partial=True)
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def dotted_overrides(items, stage=None):
    """Parse --set section.key=YAML-value; do not evaluate shell/Python text."""
    result = {}
    assigned = []
    for item in items:
        if "=" not in item:
            raise ConfigError("Expected KEY=VALUE: {}".format(item))
        key, raw = item.split("=", 1)
        parts = key.split(".")
        if not all(parts) or any(key == old or key.startswith(old + ".") or old.startswith(key + ".") for old in assigned):
            raise ConfigError("Duplicate or overlapping override: {}".format(key))
        try:
            value = yaml.load(raw, Loader=UniqueLoader)
        except yaml.YAMLError as error:
            raise ConfigError("Invalid override: {}".format(item)) from error
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        assigned.append(key)
    if stage is None:
        validate_config(result, partial=True)
    else:
        validate_job_override(result, stage)
    return result


def make_snapshot(config, sources, layers):
    validate_config(config)
    payload = {"schema_version": SCHEMA_VERSION, "config": deepcopy(config),
               "config_fingerprint": fingerprint(config), "sources": deepcopy(sources), "layers": deepcopy(layers)}
    return {**payload, "snapshot_fingerprint": fingerprint(payload)}


def validate_snapshot(snapshot):
    fields = {"schema_version", "config", "config_fingerprint", "sources", "layers", "snapshot_fingerprint"}
    if not isinstance(snapshot, dict) or set(snapshot) != fields or snapshot["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("Unsupported or malformed configuration snapshot")
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_fingerprint"}
    if fingerprint(payload) != snapshot["snapshot_fingerprint"] or fingerprint(snapshot["config"]) != snapshot["config_fingerprint"]:
        raise ConfigError("Configuration snapshot fingerprint mismatch")
    validate_config(snapshot["config"])
    if (not isinstance(snapshot["layers"], dict) or not isinstance(snapshot["sources"], list)
            or "cli" not in snapshot["layers"]):
        raise ConfigError("Missing configuration provenance")
    validate_config(snapshot["layers"]["cli"], partial=True)
    return snapshot


def load_config(setting_path, default_path=DEFAULT_CONFIG, cli_overrides=None):
    defaults, default_source = _load_yaml(DEFAULT_CONFIG if default_path is None else default_path)
    setting, setting_source = _load_yaml(setting_path)
    if "setting" not in setting.get("experiment", {}) or "name" not in setting.get("experiment", {}):
        raise ConfigError("Setting YAML must explicitly name experiment.name and experiment.setting")
    cli = {} if cli_overrides is None else deepcopy(cli_overrides)
    validate_config(cli, partial=True)
    config = deep_merge(deep_merge(defaults, setting), cli)
    return make_snapshot(config, [default_source, setting_source], {"cli": cli})


def resolve_job_config(snapshot, stage, job_id, job_override=None, cli_override=None):
    """Resolve a low-level job without allowing data/model lineage to be overridden."""
    validate_snapshot(snapshot)
    require_id(job_id)
    if stage not in STAGES or stage not in snapshot["config"]["experiment"]["enabled_stages"]:
        raise ConfigError("Stage is not enabled: {}".format(stage))
    config = deepcopy(snapshot["config"])
    configured = config[stage]["jobs"].get(job_id, {})
    explicit = {} if job_override is None else deepcopy(job_override)
    cli = {} if cli_override is None else deepcopy(cli_override)
    validate_job_override(explicit, stage)
    validate_job_override(cli, stage)
    # CLI precedence: stage-wide, then job-specific, then low-level job CLI.
    stage_cli = snapshot["layers"]["cli"].get(stage, {})
    inherited_cli = {key: value for key, value in stage_cli.items() if key in ("params", "run")}
    inherited_job_cli = stage_cli.get("jobs", {}).get(job_id, {})
    for override in (configured, explicit, inherited_cli, inherited_job_cli, cli):
        config[stage] = deep_merge(config[stage], override)
    layers = {**snapshot["layers"], "parent_snapshot": snapshot["snapshot_fingerprint"],
              "job_id": job_id, "stage": stage, "job": deep_merge(configured, explicit),
              "high_level_job_cli": deepcopy(inherited_job_cli), "job_cli": cli}
    return make_snapshot(config, snapshot["sources"], layers)



def resolve_run_config(snapshot, stage, cli_override=None):
    """Derive one high-level run from the frozen experiment, without reloading YAML."""
    validate_snapshot(snapshot)
    if stage not in STAGES or stage not in snapshot["config"]["experiment"]["enabled_stages"]:
        raise ConfigError("Stage is not enabled: {}".format(stage))
    cli = {} if cli_override is None else deepcopy(cli_override)
    validate_job_override(cli, stage)
    config = deep_merge(snapshot["config"], {stage: cli})
    layers = {**snapshot["layers"], "parent_snapshot": snapshot["snapshot_fingerprint"], "stage": stage,
              "cli": deep_merge(snapshot["layers"]["cli"], {stage: cli})}
    return make_snapshot(config, snapshot["sources"], layers)
