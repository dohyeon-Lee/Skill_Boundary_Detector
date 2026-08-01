#!/usr/bin/env bash
# 디스크 사용량 확인 — 실행 위치를 자동 감지해서 해당 서버/디스크만 측정
#
# 스크립트가 놓인 경로 기준으로 자동 결정됨:
#   /data1/dohyeon/SBD 에서 실행   → /data1 디스크 + /data1/dohyeon 순위 + 워크스페이스 세부
#   /data2/dohyeon/SBD 에서 실행   → /data2 기준
#   /scratch(2)/mdorazi/SBD 에서 실행 → 해당 scratch 기준
#
# 사용법:
#   ./check_disk_usage.sh              # 기본 (디스크 현황 + 사용자 폴더 순위 + 워크스페이스 세부)
#   DEPTH=2 ./check_disk_usage.sh      # 워크스페이스 세부를 2단계까지
#   DIR=outputs ./check_disk_usage.sh  # 워크스페이스에서 outputs 만
#   MIN_SIZE=1G ./check_disk_usage.sh  # 1G 미만 항목 숨기기
#
# 환경변수:
#   ROOT            워크스페이스 경로 (기본: 스크립트가 있는 폴더)
#   BASE            사용자 폴더 경로 (기본: ROOT의 부모, e.g. /data1/dohyeon)
#   DEPTH           워크스페이스 하위 폴더 탐색 깊이 (기본: 1)
#   DIR             워크스페이스에서 특정 폴더만 (비워두면 전체)
#   MIN_SIZE        이 값보다 작은 항목 숨기기, e.g. 1G (기본: 0, 전체 표시)
#   INCLUDE_HIDDEN  사용자 폴더 순위에 숨김 폴더 포함 여부: 1 포함, 0 제외 (기본: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${ROOT:-${SCRIPT_DIR}}"
BASE="${BASE:-$(dirname "${ROOT}")}"
DEPTH="${DEPTH:-1}"
DIR="${DIR:-}"
MIN_SIZE="${MIN_SIZE:-0}"
INCLUDE_HIDDEN="${INCLUDE_HIDDEN:-1}"

bytes_min=0
if [[ "$MIN_SIZE" != "0" ]]; then
    bytes_min=$(numfmt --from=iec "$MIN_SIZE" 2>/dev/null || echo 0)
fi

# MIN_SIZE 필터: 0 리턴이면 표시, 1이면 숨김
too_small() {
    local path="$1" bytes_sz
    (( bytes_min > 0 )) || return 1
    bytes_sz=$(du -sb "$path" 2>/dev/null | cut -f1)
    (( bytes_sz < bytes_min ))
}

# ── 전체 디스크 현황 ─────────────────────────────────────────────────────────
echo "============================  디스크 현황  ============================"
echo "  서버: $(hostname)    마운트: $(df -T "$ROOT" | awk 'NR==2{print $1, "["$2"]"}')"
df -h "$ROOT" | sed -n '1,2p'
echo

# ── 사용자 폴더(BASE) 상위 폴더 용량 순위 ────────────────────────────────────
BASE_TARGETS=()
if [[ "$INCLUDE_HIDDEN" == "1" ]]; then
    mapfile -t BASE_TARGETS < <(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort)
else
    mapfile -t BASE_TARGETS < <(find "$BASE" -mindepth 1 -maxdepth 1 -type d -not -name '.*' | sort)
fi

if (( ${#BASE_TARGETS[@]} > 0 )); then
    echo "===================  ${BASE} 상위 폴더 용량 순위  ==================="
    printf "  %-6s  %-12s  %s\n" "순위" "용량" "폴더"
    printf "  %-6s  %-12s  %s\n" "----" "----" "----"

    rank=1
    mapfile -t ranked < <(du -sh "${BASE_TARGETS[@]}" 2>/dev/null | sort -rh)
    for line in "${ranked[@]}"; do
        sz="${line%%$'\t'*}"
        path="${line#*$'\t'}"
        too_small "$path" && continue
        printf "  %2d      %-12s  %s\n" "$rank" "$sz" "${path#${BASE}/}"
        ((rank++))
    done

    echo
    printf "  합계: "
    du -sh "$BASE" 2>/dev/null | cut -f1
    echo
fi

# ── 워크스페이스(ROOT) 폴더별 세부 용량 ──────────────────────────────────────
TARGETS=()
if [[ -z "$DIR" ]]; then
    mapfile -t TARGETS < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -not -name '.*' | sort)
else
    TARGETS=("${ROOT}/${DIR}")
fi

if (( ${#TARGETS[@]} == 0 )); then
    echo "대상 폴더 없음: ${ROOT}"
    exit 1
fi

# 상위 폴더 du 한 번만 계산 후 용량순 정렬
mapfile -t ranked < <(du -sh "${TARGETS[@]}" 2>/dev/null | sort -rh)

declare -A TOTAL
SORTED=()
for line in "${ranked[@]}"; do
    sz="${line%%$'\t'*}"
    path="${line#*$'\t'}"
    SORTED+=("$path")
    TOTAL["$path"]="$sz"
done

echo "===================  워크스페이스 용량 순위 (${ROOT})  ==================="
rank=1
for target in "${SORTED[@]}"; do
    printf "  %2d.  %-10s  %s\n" "$rank" "${TOTAL[$target]}" "${target#${ROOT}/}"
    ((rank++))
done
echo

for target in "${SORTED[@]}"; do
    echo "════════════════════════════════════════════════════════════════════"
    printf "  %-60s  %s\n" "${target}/" "[합계: ${TOTAL[$target]}]"
    echo "════════════════════════════════════════════════════════════════════"

    mapfile -t dirs < <(
        find "$target" -mindepth 1 -maxdepth "$DEPTH" -type d | sort
    )

    if (( ${#dirs[@]} == 0 )); then
        echo "  (하위 폴더 없음)"
        echo
        continue
    fi

    sizes=$(du -sh "${dirs[@]}" 2>/dev/null | sort -rh)

    while IFS=$'\t' read -r sz path; do
        too_small "$path" && continue
        printf "  %-12s  %s\n" "$sz" "${path#${ROOT}/}"
    done <<< "$sizes"

    echo
done
