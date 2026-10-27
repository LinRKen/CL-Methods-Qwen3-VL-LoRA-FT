"""Explicit file/directory manifests shared by tune, infer and eval."""

from copy import deepcopy
from pathlib import Path
import os

from .common import SCHEMA_VERSION, STAGES, fingerprint, require_id, sha256_file

ARTIFACT_KINDS = ("base_model", "judge_model", "processor", "adapter", "checkpoint", "annotation",
                  "prediction", "judge_raw", "judge_scored", "metrics", "summary")


def _file_identity(path):
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Artifact changed while hashing: {}".format(path))
    return {"size": after.st_size, "sha256": digest}


def path_identity(path):
    """Hash contents, including model/adapter weights; callers may reuse a manifest."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file():
        return {"type": "file", **_file_identity(path)}
    if not path.is_dir():
        raise ValueError("Artifact is not a regular file or directory: {}".format(path))
    files = []
    for directory, subdirs, names in os.walk(path, followlinks=False):
        subdirs.sort()
        for name in subdirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Directory symlinks inside an artifact are not supported: {}".format(Path(directory) / name))
        for name in sorted(names):
            item = Path(directory) / name
            if not item.is_file():
                raise ValueError("Artifact contains a non-file or broken symlink: {}".format(item))
            entry = {"name": item.relative_to(path).as_posix(), **_file_identity(item)}
            if item.is_symlink():
                entry["symlink_target"] = str(item.resolve(strict=True))
            files.append(entry)
    if not files:
        raise ValueError("Artifact directory is empty: {}".format(path))
    return {"type": "directory", "files": sorted(files, key=lambda item: item["name"]), "hash_method": "sha256-all-files"}


def _check_kind_path(kind, path):
    directory_kinds = ("base_model", "judge_model", "processor", "adapter", "checkpoint")
    if kind in directory_kinds:
        if not path.is_dir():
            raise ValueError("{} artifact must be an exact directory: {}".format(kind, path))
    elif not path.is_file():
        raise ValueError("{} artifact must be a file: {}".format(kind, path))
    if kind in ("adapter", "checkpoint"):
        if (not (path / "adapter_config.json").is_file()
                or not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))):
            raise ValueError("Adapter manifest must point directly to an adapter directory: {}".format(path))
    if kind == "checkpoint" and not (path / "trainer_state.json").is_file():
        raise ValueError("Training-state checkpoint requires trainer_state.json")
    if kind in ("base_model", "judge_model") and not (path / "config.json").is_file():
        raise ValueError("Model directory lacks config.json")


def _check_provenance(producer, source, metadata):
    if (producer is None) == (source is None):
        raise ValueError("Specify exactly one producer or external source description")
    if producer is not None:
        fields = {"experiment_id", "stage", "run_id", "job_id"}
        if not isinstance(producer, dict) or set(producer) != fields or producer["stage"] not in STAGES:
            raise ValueError("Producer must identify experiment, stage, run and job")
        for key in fields:
            require_id(producer[key])
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("External artifacts require a non-empty source description")
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be a JSON object")


def make_manifest(kind, path, *, producer=None, source=None, metadata=None):
    """External inputs require a source description; produced artifacts identify their job."""
    if kind not in ARTIFACT_KINDS:
        raise ValueError("Unsupported artifact kind: {}".format(kind))
    metadata = {} if metadata is None else deepcopy(metadata)
    producer = deepcopy(producer)
    _check_provenance(producer, source, metadata)
    requested = str(Path(path).expanduser().absolute())
    resolved = Path(path).expanduser().resolve(strict=True)
    _check_kind_path(kind, resolved)
    payload = {"schema_version": SCHEMA_VERSION, "kind": kind, "requested_path": requested,
               "path": str(resolved), "identity": path_identity(resolved), "producer": producer,
               "external_source": source, "metadata": metadata}
    return {**payload, "fingerprint": fingerprint(payload)}


def check_manifest(manifest, verify_contents=False):
    fields = {"schema_version", "kind", "requested_path", "path", "identity", "producer",
              "external_source", "metadata", "fingerprint"}
    if (not isinstance(manifest, dict) or set(manifest) != fields
            or manifest["schema_version"] != SCHEMA_VERSION or manifest["kind"] not in ARTIFACT_KINDS):
        raise ValueError("Unsupported or malformed artifact manifest")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("Artifact manifest fingerprint mismatch")
    _check_provenance(manifest["producer"], manifest["external_source"], manifest["metadata"])
    if any(not isinstance(manifest[key], str) or not Path(manifest[key]).is_absolute()
           for key in ("path", "requested_path")):
        raise ValueError("Artifact paths must be absolute")
    path = Path(manifest["path"])
    _check_kind_path(manifest["kind"], path)
    if verify_contents and path_identity(path) != manifest["identity"]:
        raise ValueError("Artifact contents no longer match the manifest: {}".format(path))
    return manifest
