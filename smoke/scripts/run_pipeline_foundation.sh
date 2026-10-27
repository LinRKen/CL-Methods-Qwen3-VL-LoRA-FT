#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN=$(printenv PYTHON_BIN || true)
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN=/home/linruikang/data/envs/Qwen3VL-LoRA-FT/bin/python
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EXISTING_SMOKE_RESULT_DIRECTORY" >&2
  exit 2
fi
exec "$PYTHON_BIN" -B "$REPO_ROOT/smoke/scripts/test_pipeline_foundation.py" --result-dir "$1"
