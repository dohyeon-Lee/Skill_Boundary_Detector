#!/usr/bin/env bash
# wandb 로컬 로그 및 캐시 청소 — 실행 위치를 자동 감지해서 해당 워크스페이스만 청소
#
# 청소 대상 (존재하는 것만 자동 선택):
#   1. 워크스페이스 아래 모든 wandb/ 런 폴더 (run-*, offline-run-*)
#   2. 워크스페이스 내부 캐시: <워크스페이스>/.cache/wandb/{cache,data}
#   3. 환경변수 캐시: $WANDB_CACHE_DIR, $WANDB_DATA_DIR
#   4. 사용자 폴더 캐시: <워크스페이스 부모>/.wandb_cache, .wandb_data
#
# 사용법:
#   ./cleanup_wandb.sh          # 기본: 7일 이상 된 run + 30일 이상 된 캐시만 삭제
#   ./cleanup_wandb.sh --all    # 전체 삭제
#
# 환경변수:
#   BASE  워크스페이스 경로 (기본: 스크립트가 있는 폴더)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-${SCRIPT_DIR}}"
PARENT="$(dirname "${BASE}")"

ALL=false
if [[ "${1:-}" == "--all" ]]; then
  ALL=true
fi

# ── 캐시/데이터 디렉토리 후보 수집 (존재하는 것만, 중복 제거) ────────────────
CANDIDATES=(
  "${WANDB_CACHE_DIR:-}"
  "${WANDB_DATA_DIR:-}"
  "${BASE}/.cache/wandb/cache"
  "${BASE}/.cache/wandb/data"
  "${PARENT}/.wandb_cache"
  "${PARENT}/.wandb_data"
)
CACHE_DIRS=()
for d in "${CANDIDATES[@]}"; do
  [[ -n "$d" && -d "$d" ]] || continue
  dup=0
  for e in "${CACHE_DIRS[@]}"; do
    [[ "$e" == "$d" ]] && { dup=1; break; }
  done
  (( dup )) || CACHE_DIRS+=("$d")
done

# 워크스페이스 아래 wandb 런 폴더 목록 (.venv, .cache 내부 제외)
list_wandb_dirs() {
  find "${BASE}" -type d -name wandb ! -path "*/.venv/*" ! -path "*/.cache/*" 2>/dev/null || true
}

# 전체 대상 용량을 바이트로 합산 후 사람이 읽는 단위로 출력
total_size() {
  local total=0 b d
  while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    b=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
    total=$(( total + b ))
  done < <(list_wandb_dirs; printf '%s\n' "${CACHE_DIRS[@]+"${CACHE_DIRS[@]}"}")
  numfmt --to=iec "$total" 2>/dev/null || echo "${total}B"
}

echo "===== wandb 청소 시작 (서버: $(hostname), 워크스페이스: ${BASE}) ====="
echo
echo "-- 청소 대상 --"
while IFS= read -r d; do
  [[ -n "$d" ]] || continue
  du -sh "$d" 2>/dev/null || true
done < <(list_wandb_dirs; printf '%s\n' "${CACHE_DIRS[@]+"${CACHE_DIRS[@]}"}")
echo
echo "청소 전 총합: $(total_size)"
echo

if $ALL; then
  echo "[--all] 전체 삭제 중..."
  list_wandb_dirs | while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    rm -rf "$d"
  done
  for d in "${CACHE_DIRS[@]+"${CACHE_DIRS[@]}"}"; do
    rm -rf "$d"
  done
else
  echo "[기본] 7일 이상 된 run, 30일 이상 된 캐시 삭제 중..."
  list_wandb_dirs | while IFS= read -r wandb_dir; do
    [[ -n "$wandb_dir" ]] || continue
    find "${wandb_dir}" -maxdepth 1 -type d -name "run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    find "${wandb_dir}" -maxdepth 1 -type d -name "offline-run-*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
  done
  for d in "${CACHE_DIRS[@]+"${CACHE_DIRS[@]}"}"; do
    find "$d" -mindepth 2 -maxdepth 2 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true
  done
fi

echo
echo "===== 청소 완료 ====="
echo "청소 후 총합: $(total_size)"
