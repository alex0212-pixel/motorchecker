#!/usr/bin/env bash
set -euo pipefail

# RunPod 웹터미널에서 재현 가능한 환경 구축
# - 가상환경 생성
# - 의존성 설치
# - 로그/출력 디렉토리 생성

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p runpod/logs outputs

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install -U pip

# 기본 학습 스택 (Qwen2-VL + LoRA/QLoRA)
python -m pip install -U \
  "transformers>=4.45.0" \
  "accelerate>=0.34.0" \
  "datasets>=2.20.0" \
  "peft>=0.12.0" \
  "trl>=0.9.6" \
  "bitsandbytes>=0.43.0" \
  "huggingface_hub>=0.24.0" \
  "safetensors>=0.4.0" \
  "pillow>=10.0.0" \
  "tqdm>=4.66.0" \
  "albumentations>=1.4.0" \
  "opencv-python-headless>=4.10.0.84" \
  "pandas>=2.0.0"

echo "[OK] Environment ready."
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# (선택) 설치 내역 고정
python -m pip freeze > runpod/requirements.lock.txt
