#!/usr/bin/env bash
set -euo pipefail

# Shared RunPod workspace에서 사용자별 충돌 방지용 경로 규칙
# - RUNPOD_USER가 있으면 그 값을 우선 사용
# - 없으면 USER 값을 사용
# - 둘 다 없으면 "anon"

export RUNPOD_USER="${RUNPOD_USER:-${USER:-anon}}"

# 사용자별 결과물 루트
export RUNPOD_USER_DIR="${RUNPOD_USER_DIR:-outputs/users/${RUNPOD_USER}}"

# 학습 산출물(LoRA adapter) 기본 경로
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNPOD_USER_DIR}/qwen2vl-lora}"

# 학습이 끝나면 로컬로 다운로드하기 쉬운 아카이브 생성 경로
export EXPORT_DIR="${EXPORT_DIR:-${RUNPOD_USER_DIR}/exports}"
mkdir -p "$RUNPOD_USER_DIR" "$EXPORT_DIR"

echo "RUNPOD_USER=$RUNPOD_USER"
echo "RUNPOD_USER_DIR=$RUNPOD_USER_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EXPORT_DIR=$EXPORT_DIR"

