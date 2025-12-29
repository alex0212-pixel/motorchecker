#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

# user-specific default paths
source runpod/user_paths.sh

export ADAPTER_DIR="${ADAPTER_DIR:-$OUTPUT_DIR}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export MODE="${MODE:-adapter}"  # adapter|merged

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required"
  exit 1
fi

if [ -z "${HF_REPO_ID:-}" ]; then
  echo "HF_REPO_ID is required (e.g. yourname/motor-sticker-qwen2vl-lora)"
  exit 1
fi

python runpod/push_to_hf.py
