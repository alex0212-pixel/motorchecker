#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

# user-specific default paths
source runpod/user_paths.sh

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
export ADAPTER_DIR="${ADAPTER_DIR:-$OUTPUT_DIR}"
export TEST_IMAGE="${TEST_IMAGE:-data/motor_checker/20240817_000105.jpg}"
export QUESTION="${QUESTION:-스티커가 있나요? 있다면 번호(number)와 색(color: 초록색/노란색/빨간색)을 JSON으로만 답하세요.}"

python runpod/infer_qwen2vl_lora.py
