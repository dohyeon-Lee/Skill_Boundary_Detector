#!/usr/bin/env bash
# wandb 로컬 로그 및 캐시 청소 스크립트
# 사용법:
#   ./cleanup_wandb.sh          # 기본: 7일 이상 된 run만 삭제
#   ./cleanup_wandb.sh --all    # 전체 삭제

set -euo pipefail

BASE=/data2/dohyeon/SBD
CACHE_DIR=/data2/dohyeon/.wandb_cache
DATA_DIR=/data2/dohyeon/.wandb_data
ALL=false

if [[ "${1:-}" == "--all" ]]; then
  ALL=true
fi

echo "===== wandb 청소 시작 ====="
du -sh ${CACHE_DIR} ${DATA_DIR} 2>/dev/null || true
find ${BASE} -type d -name "wandb" ! -path "*/.venv/*" -exec du -sh {} \; 2>/dev/null | awk '{sum+=$1} END {print "로컬 wandb 총합: " sum "K"}' || true

if $ALL; then
  echo "[--all] 전체 삭제 중..."
  find ${BASE} -type d -name "wandb" ! -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
  rm -rf ${CACHE_DIR} ${DATA_DIR}
else
  echo "[기본] 7일 이상 된 run 삭제 중..."
  # run-YYYYMMDD_HHMMSS-* 형식의 오래된 디렉토리만 삭제
  find ${BASE} -type d -name "wandb" ! -path "*/.venv/*" | while read wandb_dir; do
    find "${wandb_dir}" -maxdepth 1 -type d -name "run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    find "${wandb_dir}" -maxdepth 1 -type d -name "offline-run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
  done
  # cache는 30일 이상 된 것만
  if [[ -d ${CACHE_DIR} ]]; then
    find ${CACHE_DIR} -mindepth 2 -maxdepth 2 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
  fi
fi

echo "===== 청소 완료 ====="
du -sh ${CACHE_DIR} ${DATA_DIR} 2>/dev/null || true
