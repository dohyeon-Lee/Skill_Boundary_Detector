#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   1) DRY_RUN=1 ./prune_checkpoints.sh           # 삭제 없이 미리보기
#   2) DRY_RUN=0 ./prune_checkpoints.sh           # 실제 삭제
#   3) OUTPUTS_DIR=... DRY_RUN=1 ./prune_checkpoints.sh
#   4) EXCLUDE=pi05_libero_10 DRY_RUN=0 ./prune_checkpoints.sh
# 옵션:
#   KEEP=5 DRY_RUN=1 ./prune_checkpoints.sh

KEEP="${KEEP:-5}"
DRY_RUN="${DRY_RUN:-1}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/data2/dohyeon/SBD/outputs}"
EXCLUDE="${EXCLUDE:-}"  # 제외할 checkpoints 경로 (부분 문자열 매칭)

echo "[INFO] KEEP=$KEEP DRY_RUN=$DRY_RUN OUTPUTS_DIR=$OUTPUTS_DIR"
[[ -n "$EXCLUDE" ]] && echo "[INFO] EXCLUDE=$EXCLUDE"
echo "[INFO] before:"
df -h /data2 | sed -n '1,2p'
echo

# outputs 폴더 아래에서 checkpoints 디렉토리를 자동으로 찾음
mapfile -t all_targets < <(
  find "$OUTPUTS_DIR" -mindepth 2 -maxdepth 2 -type d -name "checkpoints" | sort
)

if (( ${#all_targets[@]} == 0 )); then
  echo "[WARN] No checkpoints directories found under $OUTPUTS_DIR"
  exit 0
fi

# ── 용량 요약 먼저 출력 ────────────────────────────────────────────────────────
echo "=== Checkpoint size summary ==="
printf "%-12s  %s\n" "SIZE" "CHECKPOINTS DIR"
for t in "${all_targets[@]}"; do
  [[ -n "$EXCLUDE" && "$t" == *"$EXCLUDE"* ]] && continue
  sz=$(du -sh "$t" 2>/dev/null | cut -f1)
  printf "%-12s  %s\n" "$sz" "$t"
done
echo

# ── checkpoint 별 prune ───────────────────────────────────────────────────────
for t in "${all_targets[@]}"; do
  [[ -n "$EXCLUDE" && "$t" == *"$EXCLUDE"* ]] && { echo "[SKIP] $t"; echo; continue; }

  mapfile -t steps < <(
    find "$t" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]{6}$' \
      | sort
  )

  total="${#steps[@]}"
  del=$(( total > KEEP ? total - KEEP : 0 ))

  echo "=== $t ==="
  echo "total=$total keep=$KEEP delete=$del"

  if (( del == 0 )); then
    echo "nothing to delete"
    echo
    continue
  fi

  echo "delete range: ${steps[0]} ... ${steps[$((del-1))]}"
  echo "keep range  : ${steps[$del]} ... ${steps[$((total-1))]}"

  if [[ "$DRY_RUN" == "0" ]]; then
    for s in "${steps[@]:0:$del}"; do
      rm -rf -- "$t/$s"
    done
    echo "[DONE] deleted=$del"
  else
    echo "[DRY RUN] no deletion performed"
  fi

  echo "remaining:"
  ls -1 "$t" | grep -E '^[0-9]{6}$|^last$' | sort
  echo
done

echo "[INFO] after:"
df -h /data2 | sed -n '1,2p'
