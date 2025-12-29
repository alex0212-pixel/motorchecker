#!/usr/bin/env bash
set -euo pipefail

# Qwen2-VL LoRA/QLoRA 학습 실행 스크립트
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

# 사용자별 출력 경로 자동 설정(공용 workspace에서 덮어쓰기 방지)
source runpod/user_paths.sh

# ===== 사용자가 주로 바꾸는 값들 =====
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
export TRAIN_JSONL="${TRAIN_JSONL:-/workspace/train.jsonl}"
# OUTPUT_DIR is defined by user_paths.sh (can override before calling this script)

# 학습 하이퍼파라미터(기본값은 안전하게 작게)
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LR="${LR:-2e-4}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"

# QLoRA 사용 여부
export USE_QLORA="${USE_QLORA:-true}"

# augmentation (rotation/perspective etc.)
export AUGMENT="${AUGMENT:-true}"
export AUG_P="${AUG_P:-0.85}"
export AUG_MAX_ROTATE="${AUG_MAX_ROTATE:-180}"

mkdir -p "$OUTPUT_DIR" runpod/logs

echo "MODEL_NAME=$MODEL_NAME"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "USE_QLORA=$USE_QLORA"
echo "AUGMENT=$AUGMENT AUG_P=$AUG_P AUG_MAX_ROTATE=$AUG_MAX_ROTATE"
echo "NUM_EPOCHS=$NUM_EPOCHS LR=$LR BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM"

accelerate launch --mixed_precision bf16 \
  runpod/train_qwen2vl_lora.py

# 학습 결과를 로컬로 다운로드하기 쉽도록 tar.gz 아카이브 생성
bash runpod/export_artifacts.sh
