#!/usr/bin/env bash
set -euo pipefail

# RunPod 새 Pod(아무것도 없는 상태)에서 처음부터 레포 클론→환경 구축까지
#
# 사용법:
#   export REPO_URL="https://github.com/<you>/<repo>.git"   # 또는 git@...
#   export RUNPOD_USER="kimjunyoung"                       # 사용자별 출력 경로 분리(권장)
#   bash -lc "curl -fsSL <RAW_URL>/runpod/bootstrap_from_scratch.sh | bash"
#
# 이 파일은 레포 안에도 들어있어서, 레포를 받은 뒤에는:
#   bash runpod/bootstrap_from_scratch.sh
# 를 그대로 실행해도 됩니다.

REPO_URL="${REPO_URL:-}"
if [ -z "$REPO_URL" ]; then
  echo "REPO_URL env is required"
  echo "example: export REPO_URL=https://github.com/you/your-repo.git"
  exit 1
fi

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[INFO] cloning repo: $REPO_URL"
git clone "$REPO_URL"

REPO_NAME="${REPO_NAME:-$(basename "$REPO_URL" .git)}"
cd "$WORKDIR/$REPO_NAME"

echo "[OK] repo cloned: $(pwd)"

echo "[INFO] running setup"
bash runpod/setup.sh

echo "[OK] bootstrap done"
echo
echo "Next:"
echo "  export RUNPOD_USER=your_name_or_id"
echo "  # 라벨 생성→채우기→jsonl 생성 후"
echo "  bash runpod/train_qwen2vl_lora.sh"

