#!/usr/bin/env bash
# Skill_Boundary_Detector 폴더 내부 사용량만 확인한다.
# 부모 사용자 폴더나 전체 마운트 사용량은 조회하지 않는다.
#
# 사용법:
#   ./check_disk_usage.sh              # SBD 바로 아래 항목을 한 번씩 측정
#   DEPTH=1 ./check_disk_usage.sh      # 각 항목의 하위 폴더도 1단계 표시
#   DIR=outputs ./check_disk_usage.sh  # SBD 내부의 특정 항목만
#   MIN_SIZE=1G ./check_disk_usage.sh  # 1G 미만 항목 숨기기
#
# 환경변수:
#   DEPTH     하위 폴더 표시 깊이 (기본: 0, 최상위 사용량만)
#   DIR       SBD 기준 상대 경로 (비워두면 바로 아래 모든 항목)
#   MIN_SIZE  이 값보다 작은 항목 숨기기, e.g. 1G (기본: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH="${DEPTH:-0}"
DIR="${DIR:-}"
MIN_SIZE="${MIN_SIZE:-0}"

if ! [[ "${DEPTH}" =~ ^[0-9]+$ ]]; then
    echo "DEPTH는 0 이상의 정수여야 합니다: ${DEPTH}" >&2
    exit 2
fi

bytes_min=0
if [[ "${MIN_SIZE}" != "0" ]]; then
    if ! bytes_min=$(numfmt --from=iec "${MIN_SIZE}" 2>/dev/null); then
        echo "잘못된 MIN_SIZE 값입니다: ${MIN_SIZE}" >&2
        exit 2
    fi
fi

relative_path() {
    local path="$1"
    if [[ "${path}" == "${ROOT}" ]]; then
        printf '.'
    else
        printf '%s' "${path#${ROOT}/}"
    fi
}

human_kib() {
    numfmt --to=iec --from-unit=1024 "$1"
}

large_enough() {
    local kib="$1"
    (( bytes_min == 0 || kib * 1024 >= bytes_min ))
}

TARGETS=()
if [[ -z "${DIR}" ]]; then
    # 숨김 항목과 루트 파일도 실제 SBD 사용량에 포함한다.
    mapfile -d '' -t TARGETS < <(
        find "${ROOT}" -mindepth 1 -maxdepth 1 -print0 | sort -z
    )
else
    requested="${ROOT}/${DIR}"
    if [[ ! -e "${requested}" ]]; then
        echo "대상 항목이 없습니다: ${requested}" >&2
        exit 1
    fi
    resolved=$(realpath -- "${requested}")
    case "${resolved}" in
        "${ROOT}"|"${ROOT}"/*) ;;
        *)
            echo "DIR은 Skill_Boundary_Detector 내부 경로여야 합니다: ${DIR}" >&2
            exit 2
            ;;
    esac
    TARGETS=("${resolved}")
fi

if (( ${#TARGETS[@]} == 0 )); then
    echo "대상 항목 없음: ${ROOT}" >&2
    exit 1
fi

echo "================ Skill_Boundary_Detector 내부 사용량 ================"
echo "  경로: ${ROOT}"
[[ -z "${DIR}" ]] || echo "  선택: ${DIR}"
echo "  대상: ${#TARGETS[@]}개 · 상세 깊이: ${DEPTH}"
echo "  각 항목은 측정이 끝나는 즉시 표시됩니다."
echo

records=()
total_kib=0
measured=0
target_count=${#TARGETS[@]}

for target in "${TARGETS[@]}"; do
    ((++measured))
    label=$(relative_path "${target}")
    printf '[%d/%d] %-48s 측정 중... ' "${measured}" "${target_count}" "${label}"

    # --max-depth를 이용해 상위 합계와 요청된 상세를 한 번의 순회로 얻는다.
    mapfile -t usage_lines < <(
        du -x -k --max-depth="${DEPTH}" -- "${target}" 2>/dev/null
    )

    root_kib=""
    detail_records=()
    for line in "${usage_lines[@]}"; do
        kib="${line%%$'\t'*}"
        path="${line#*$'\t'}"
        if [[ "${path}" == "${target}" ]]; then
            root_kib="${kib}"
        else
            detail_records+=("${kib}"$'\t'"${path}")
        fi
    done

    if [[ -z "${root_kib}" ]]; then
        echo "실패"
        continue
    fi

    ((total_kib += root_kib))
    size=$(human_kib "${root_kib}")
    echo "${size}"

    if large_enough "${root_kib}"; then
        records+=("${root_kib}"$'\t'"${target}")
    fi

    if (( DEPTH > 0 && ${#detail_records[@]} > 0 )); then
        mapfile -t sorted_details < <(
            printf '%s\n' "${detail_records[@]}" | sort -t $'\t' -k1,1nr
        )
        for detail in "${sorted_details[@]}"; do
            kib="${detail%%$'\t'*}"
            path="${detail#*$'\t'}"
            large_enough "${kib}" || continue
            printf '         %-10s  %s\n' "$(human_kib "${kib}")" "$(relative_path "${path}")"
        done
    fi
done

echo
echo "  측정 합계: $(human_kib "${total_kib}")"
echo

if (( ${#records[@]} == 0 )); then
    echo "MIN_SIZE=${MIN_SIZE} 이상인 항목이 없습니다."
    exit 0
fi

mapfile -t ranked < <(
    printf '%s\n' "${records[@]}" | sort -t $'\t' -k1,1nr
)

echo "========================= 상위 항목 용량 순위 ========================="
rank=1
for line in "${ranked[@]}"; do
    kib="${line%%$'\t'*}"
    path="${line#*$'\t'}"
    printf '  %2d.  %-10s  %s\n' \
        "${rank}" "$(human_kib "${kib}")" "$(relative_path "${path}")"
    ((++rank))
done
