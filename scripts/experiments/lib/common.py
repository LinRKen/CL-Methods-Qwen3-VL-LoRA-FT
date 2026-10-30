"""Small JSON/filesystem helpers; never import CUDA or model libraries."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

SCHEMA_VERSION = 1
STAGES = ("tune", "infer", "eval")
TASKS = ("motion", "hypo", "pattern", "nav", "view", "geo")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def require_id(value):
    if (not isinstance(value, str) or value in (".", "..")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value)):
        raise ValueError("ID must be 1-96 letters, digits, dots, underscores or hyphens")
    return value


def new_id(prefix):
    require_id(prefix)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return require_id("{}-{}-{}".format(prefix[:60], stamp, uuid.uuid4().hex[:8]))


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON value: {}".format(value))


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".{}-".format(path.name), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def record_lock(directory):
    """Serialize record updates on the local POSIX filesystem, not GPU jobs."""
    with (Path(directory) / ".record.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
