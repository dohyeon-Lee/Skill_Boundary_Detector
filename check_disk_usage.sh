#!/usr/bin/env bash
# 디스크 사용량 확인 — outputs / libero_dataset / libero_small_dataset 폴더별 요약
#
# 사용법:
#   ./check_disk_usage.sh              # 기본 (1단계 하위폴더)
#   DEPTH=2 ./check_disk_usage.sh      # 2단계까지
#   DIR=outputs ./check_disk_usage.sh  # outputs 만
#
# 환경변수:
#   ROOT      기준 경로 (기본: /data2/dohyeon/SBD)
#   DEPTH     하위 폴더 탐색 깊이 (기본: 1)
#   DIR       특정 대상만 (outputs | libero_dataset | libero_small_dataset | 비워두면 전체)
#   MIN_SIZE  이 값보다 작은 항목 숨기기, e.g. 1G (기본: 0, 전체 표시)

set -euo pipefail

ROOT="${ROOT:-/scratch/mdorazi/Skill_Boundary_Detector}"
DEPTH="${DEPTH:-1}"
DIR="${DIR:-}"
MIN_SIZE="${MIN_SIZE:-0}"

TARGETS=()
if [[ -z "$DIR" ]]; then
    for d in outputs libero_dataset libero_small_dataset; do
        [[ -d "${ROOT}/${d}" ]] && TARGETS+=("${ROOT}/${d}")
    done
else
    TARGETS=("${ROOT}/${DIR}")
fi

if (( ${#TARGETS[@]} == 0 )); then
    echo "대상 폴더 없음: ${ROOT}"
    exit 1
fi

# ── 전체 디스크 현황 ─────────────────────────────────────────────────────────
echo "============================  디스크 현황  ============================"
df -h /scratch/mdorazi | sed -n '1,2p'
echo

# ── 폴더별 용량 출력 ─────────────────────────────────────────────────────────
for target in "${TARGETS[@]}"; do
    total=$(du -sh "$target" 2>/dev/null | cut -f1)
    echo "════════════════════════════════════════════════════════════════════"
    printf "  %-60s  %s\n" "${target}/" "[합계: ${total}]"
    echo "════════════════════════════════════════════════════════════════════"

    # DEPTH 단계까지 하위 폴더 크기 수집
    mapfile -t dirs < <(
        find "$target" -mindepth 1 -maxdepth "$DEPTH" -type d | sort
    )

    if (( ${#dirs[@]} == 0 )); then
        echo "  (하위 폴더 없음)"
        echo
        continue
    fi

    # du 한 번에 돌리고 정렬
    sizes=$(du -sh "${dirs[@]}" 2>/dev/null | sort -rh)

    while IFS=$'\t' read -r sz path; do
        # MIN_SIZE 필터 (numfmt 없는 환경 대비 간단 비교)
        if [[ "$MIN_SIZE" != "0" ]]; then
            bytes_sz=$(du -sb "$path" 2>/dev/null | cut -f1)
            bytes_min=$(numfmt --from=iec "$MIN_SIZE" 2>/dev/null || echo 0)
            (( bytes_sz < bytes_min )) && continue
        fi
        # ROOT 기준 상대경로로 표시
        rel="${path#${ROOT}/}"
        printf "  %-12s  %s\n" "$sz" "$rel"
    done <<< "$sizes"

    echo
done

echo "============================  요약  ============================"
for target in "${TARGETS[@]}"; do
    sz=$(du -sh "$target" 2>/dev/null | cut -f1)
    rel="${target#${ROOT}/}"
    printf "  %-12s  %s\n" "$sz" "$rel"
done
echo
df -h /scratch/mdorazi | sed -n '1,2p'
