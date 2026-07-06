#!/usr/bin/env bash
# /scratch/mdorazi 및 /scratch2/mdorazi 바로 아래 상위 폴더별 디스크 사용량 요약
#
# 사용법:
#   ./check_mdorazi_disk_usage.sh
#   MIN_SIZE=1G ./check_mdorazi_disk_usage.sh      # 1G 미만 숨기기
#   INCLUDE_HIDDEN=0 ./check_mdorazi_disk_usage.sh # 숨김 폴더 제외
#
# 환경변수:
#   MIN_SIZE        이 값보다 작은 폴더 숨기기, e.g. 1G (기본: 0, 전체 표시)
#   INCLUDE_HIDDEN  숨김 폴더 포함 여부: 1 포함, 0 제외 (기본: 1)

set -euo pipefail

MIN_SIZE="${MIN_SIZE:-0}"
INCLUDE_HIDDEN="${INCLUDE_HIDDEN:-1}"

ROOTS=(/scratch/mdorazi /scratch2/mdorazi)

bytes_min=0
if [[ "$MIN_SIZE" != "0" ]]; then
    bytes_min=$(numfmt --from=iec "$MIN_SIZE" 2>/dev/null || echo 0)
fi

_show_root() {
    local ROOT="$1"

    if [[ ! -d "$ROOT" ]]; then
        echo "경로 없음: ${ROOT}" >&2
        return
    fi

    TARGETS=()
    if [[ "$INCLUDE_HIDDEN" == "1" ]]; then
        while IFS= read -r d; do
            TARGETS+=("$d")
        done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
    else
        while IFS= read -r d; do
            TARGETS+=("$d")
        done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -not -name '.*' | sort)
    fi

    echo "========================================================================"
    echo "  마운트: $(df -T "$ROOT" | awk 'NR==2{print $1, "["$2"]"}')"
    df -h "$ROOT" | sed -n '1,2p'
    echo

    if (( ${#TARGETS[@]} == 0 )); then
        echo "  (하위 폴더 없음)"
        echo
        return
    fi

    printf "  %-6s  %-12s  %s\n" "순위" "용량" "폴더"
    printf "  %-6s  %-12s  %s\n" "----" "----" "----"

    rank=1
    mapfile -t ranked < <(du -sh "${TARGETS[@]}" 2>/dev/null | sort -rh)

    for line in "${ranked[@]}"; do
        sz="${line%%$'\t'*}"
        path="${line#*$'\t'}"

        if (( bytes_min > 0 )); then
            bytes_sz=$(du -sb "$path" 2>/dev/null | cut -f1 || echo 0)
            (( bytes_sz < bytes_min )) && continue
        fi

        rel="${path#${ROOT}/}"
        printf "  %2d      %-12s  %s\n" "$rank" "$sz" "$rel"
        ((rank++))
    done

    echo
    printf "  합계: "
    du -sh "$ROOT" 2>/dev/null | cut -f1 || true
    echo
}

for root in "${ROOTS[@]}"; do
    _show_root "$root"
done
