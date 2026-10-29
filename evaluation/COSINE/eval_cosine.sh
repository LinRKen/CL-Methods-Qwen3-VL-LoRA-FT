#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
: "${INPUT_FILE:?Set INPUT_FILE to an existing full infer_cosine.py JSONL}"
DATASET="${DATASET:-cosine_motion_test}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/linruikang/data/checkpoints/Qwen3-VL-4B-Instruct}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1}}"
EVAL_PYTHON="${EVAL_PYTHON:-/home/linruikang/data/envs/Qwen3VL-vLLM/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/linruikang/data/results/CL-Methods/Qwen3-VL/LoRA-FT/legacy-eval/COSINE/${DATASET}/$(date +%Y%m%d-%H%M%S)}"
# The Python entry verifies physical identity before importing CUDA libraries.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${REPO_ROOT}/qwen-vl-finetune${PYTHONPATH:+:${PYTHONPATH}}"
exec "${EVAL_PYTHON}" "${SCRIPT_DIR}/eval_cosine.py" \
    --dataset "${DATASET}" --input-file "${INPUT_FILE}" --output-dir "${OUTPUT_DIR}" \
    --judge-model "${JUDGE_MODEL}" --batch-size "${BATCH_SIZE:-4}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --repair-attempts "${REPAIR_ATTEMPTS:-1}" --dtype "${DTYPE:-bfloat16}" --seed "${SEED:-42}" "$@"
