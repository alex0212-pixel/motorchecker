#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source runpod/user_paths.sh

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="qwen2vl_lora_${RUNPOD_USER}_${TS}.tar.gz"
ARCHIVE_PATH="${EXPORT_DIR}/${ARCHIVE_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "OUTPUT_DIR not found: $OUTPUT_DIR"
  echo "Did you finish training?"
  exit 1
fi

echo "[INFO] creating archive: $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_DIR" .

echo "[OK] export done"
echo "Archive: $ARCHIVE_PATH"
echo
echo "다운로드 방법(예시):"
echo "- RunPod UI의 File Browser에서 위 tar.gz를 로컬로 다운로드"
echo "- 또는 scp/rsync로 가져오기(SSH 연결 가능한 경우)"

