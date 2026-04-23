#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   1) DRY_RUN=1 ./prune_checkpoints.sh   # 삭제 없이 미리보기
#   2) DRY_RUN=0 ./prune_checkpoints.sh   # 실제 삭제
#
# 옵션:
#   KEEP=5 DRY_RUN=1 ./prune_checkpoints.sh
#   TARGETS를 원하는 경로로 수정 가능

KEEP="${KEEP:-5}"
DRY_RUN="${DRY_RUN:-1}"

TARGETS=(
  "/data2/dohyeon/SBD/outputs/pi05_libero_10/checkpoints"
  "/data2/dohyeon/SBD/outputs/pi05_libero_spatial_object065000_libero_10_FT/checkpoints"
  "/data2/dohyeon/SBD/outputs/pi05_libero_spatial_object065000_libero_10_option1_50_FT/checkpoints"
)

echo "[INFO] KEEP=$KEEP DRY_RUN=$DRY_RUN"
echo "[INFO] before:"
df -h /data2 | sed -n '1,2p'
echo

for t in "${TARGETS[@]}"; do
  if [[ ! -d "$t" ]]; then
    echo "[WARN] skip (not found): $t"
    continue
  fi

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
