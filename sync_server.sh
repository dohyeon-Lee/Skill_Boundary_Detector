#!/usr/bin/env bash
# Sync ignored dataset/model/output folders between SBD workspaces.
#
# 실행 위치에 따라 엔드포인트가 달라짐 (yonsei→rllab 방향 접속 불가):
#   yonsei 클러스터에서 실행:  [1] /scratch  [2] /scratch2       (내부 전용)
#   rllab  클러스터에서 실행:  [1] rllab(스크립트 위치)  [2] yonsei /scratch  [3] yonsei /scratch2
#
# Usage:
#   ./sync_server.sh
#   YONSEI_HOST=user@host RLLAB_BASE=/path/to/SBD ./sync_server.sh
#   SYNC_MODE=rllab ./sync_server.sh   # hostname 감지 무시하고 모드 강제
#
# Flow:
#   1. FROM 엔드포인트, TO 엔드포인트 선택.
#   2. FROM 쪽에서 large-data/output .gitignore 대상 최상위 폴더 선택,
#      전체 보내기 or 하위폴더 드릴다운.
#   3. TO base 아래 같은 상대 경로로 rsync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YONSEI_HOST="${YONSEI_HOST:-mdorazi@165.132.142.207}"
YONSEI_SCRATCH_BASE="${YONSEI_SCRATCH_BASE:-/scratch/mdorazi/Skill_Boundary_Detector}"
YONSEI_SCRATCH2_BASE="${YONSEI_SCRATCH2_BASE:-/scratch2/mdorazi/Skill_Boundary_Detector}"
RLLAB_BASE="${RLLAB_BASE:-${SCRIPT_DIR}}"

if [ -z "${SYNC_MODE:-}" ]; then
    case "$(hostname)" in
        yonsei*) SYNC_MODE="yonsei" ;;
        *)       SYNC_MODE="rllab" ;;
    esac
fi

# host가 빈 문자열이면 로컬 경로, 아니면 ssh 대상
if [ "${SYNC_MODE}" = "yonsei" ]; then
    EP_NAMES=("yonsei /scratch" "yonsei /scratch2")
    EP_HOSTS=("" "")
    EP_BASES=("${YONSEI_SCRATCH_BASE}" "${YONSEI_SCRATCH2_BASE}")
else
    EP_NAMES=("rllab" "yonsei /scratch" "yonsei /scratch2")
    EP_HOSTS=("" "${YONSEI_HOST}" "${YONSEI_HOST}")
    EP_BASES=("${RLLAB_BASE}" "${YONSEI_SCRATCH_BASE}" "${YONSEI_SCRATCH2_BASE}")
fi

die() {
    echo "Error: $*" >&2
    exit 1
}

remote_quote() {
    printf '%q' "$1"
}

SYNC_ROOT_PATTERNS=(
    "dataset"
    "dataset_*"
    "libero_dataset"
    "libero_small_dataset"
    "libero_10_old"
    "libero_original_dataset"
    "libero_dataset_prev"
    "models"
    "outputs"
    "outputs_*"
    "outputs_DP"
    "outputs_FSQ"
    "VLA_outputs"
    "VLA_single_outputs"
    "DP_outputs"
    "FSQ_outputs"
    "pi05_outputs"
    "skillVLA_outputs"
)

print_menu() {
    local title="$1"
    shift
    local items=("$@")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${title}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for i in "${!items[@]}"; do
        printf "  [%d] %s\n" "$((i + 1))" "${items[$i]}"
    done
    echo "  [q] cancel"
    echo ""
}

choose_index() {
    local count="$1"
    local input
    while true; do
        read -r -p "선택: " input
        case "${input}" in
            q|Q|quit|exit) return 1 ;;
        esac
        if [[ "${input}" =~ ^[0-9]+$ ]] && [ "${input}" -ge 1 ] && [ "${input}" -le "${count}" ]; then
            echo "$((input - 1))"
            return 0
        fi
        echo "잘못된 선택입니다. 1-${count} 또는 q를 입력하세요." >&2
    done
}

ep_label() {
    local i="$1"
    if [ -n "${EP_HOSTS[$i]}" ]; then
        echo "${EP_NAMES[$i]} (${EP_HOSTS[$i]}:${EP_BASES[$i]})"
    else
        echo "${EP_NAMES[$i]} (${EP_BASES[$i]})"
    fi
}

# --- FROM 쪽 폴더 나열 (로컬/원격 공용) ---------------------------------

src_list_top_dirs() {
    if [ -z "${SRC_HOST}" ]; then
        local dirs=() pattern path name existing existing_name
        for pattern in "${SYNC_ROOT_PATTERNS[@]}"; do
            if [[ "${pattern}" == *'*'* ]]; then
                while IFS= read -r path; do
                    [ -d "${path}" ] || continue
                    name="$(basename "${path}")"
                    existing=0
                    for existing_name in "${dirs[@]}"; do
                        [ "${existing_name}" = "${name}" ] && { existing=1; break; }
                    done
                    [ "${existing}" -eq 1 ] || dirs+=("${name}")
                done < <(compgen -G "${SRC_BASE}/${pattern}" || true)
            else
                path="${SRC_BASE}/${pattern}"
                [ -d "${path}" ] || continue
                name="${pattern}"
                existing=0
                for existing_name in "${dirs[@]}"; do
                    [ "${existing_name}" = "${name}" ] && { existing=1; break; }
                done
                [ "${existing}" -eq 1 ] || dirs+=("${name}")
            fi
        done
        (( ${#dirs[@]} == 0 )) || printf '%s\n' "${dirs[@]}"
    else
        local quoted patterns
        quoted="$(remote_quote "${SRC_BASE}")"
        patterns="$(printf '%q ' "${SYNC_ROOT_PATTERNS[@]}")"
        ssh "${SRC_HOST}" "cd ${quoted} 2>/dev/null && for pattern in ${patterns}; do for d in \$pattern; do [ -d \"\$d\" ] && printf '%s\n' \"\$d\"; done; done | awk '!seen[\$0]++'"
    fi
}

src_list_child_dirs() {
    local rel="$1"
    if [ -z "${SRC_HOST}" ]; then
        local abs="${SRC_BASE}/${rel}"
        [ -d "${abs}" ] || return 0
        find "${abs}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    else
        local quoted
        quoted="$(remote_quote "${SRC_BASE}/${rel}")"
        ssh "${SRC_HOST}" "find ${quoted} -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort"
    fi
}

src_dir_exists() {
    local rel="$1"
    if [ -z "${SRC_HOST}" ]; then
        [ -d "${SRC_BASE}/${rel}" ]
    else
        ssh "${SRC_HOST}" "test -d $(remote_quote "${SRC_BASE}/${rel}")"
    fi
}

choose_folder() {
    local top_dirs=()
    mapfile -t top_dirs < <(src_list_top_dirs)
    [ "${#top_dirs[@]}" -gt 0 ] || die "No output/dataset folders found under ${SRC_LABEL}"

    print_menu "보낼 최상위 폴더 선택 (${SRC_LABEL})" "${top_dirs[@]}"
    local idx
    idx="$(choose_index "${#top_dirs[@]}")" || return 1

    local rel="${top_dirs[$idx]}"
    while true; do
        src_dir_exists "${rel}" || die "Source folder not found: ${SRC_LABEL}/${rel}"

        echo ""
        echo "현재 선택: ${rel}"
        echo "  [1] 이 폴더 전체 보내기"
        echo "  [2] 하위폴더 고르기"
        echo "  [q] cancel"
        echo ""

        local action
        read -r -p "선택: " action
        case "${action}" in
            1) CHOSEN_REL="${rel}"; return 0 ;;
            2)
                local children=()
                mapfile -t children < <(src_list_child_dirs "${rel}")
                if [ "${#children[@]}" -eq 0 ]; then
                    echo "하위폴더가 없습니다. 현재 폴더를 선택합니다."
                    CHOSEN_REL="${rel}"
                    return 0
                fi
                print_menu "${rel} 하위폴더 선택" "${children[@]}"
                local child_idx
                child_idx="$(choose_index "${#children[@]}")" || return 1
                rel="${rel}/${children[$child_idx]}"
                ;;
            q|Q|quit|exit) return 1 ;;
            *) echo "잘못된 선택입니다." >&2 ;;
        esac
    done
}

sync_folder() {
    local rel="$1"
    local src_abs="${SRC_BASE}/${rel}"
    local dst_abs="${DST_BASE}/${rel}"
    local src dst

    if [ -z "${SRC_HOST}" ]; then src="${src_abs}"; else src="${SRC_HOST}:${src_abs}"; fi
    if [ -z "${DST_HOST}" ]; then dst="${dst_abs}"; else dst="${DST_HOST}:${dst_abs}"; fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Sync"
    echo "  from: ${src}"
    echo "  to  : ${dst}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local confirm
    read -r -p "진행할까요? [y/N]: " confirm
    case "${confirm}" in
        y|Y|yes) ;;
        *) echo "취소했습니다."; exit 0 ;;
    esac

    if [ -n "${SRC_HOST}" ] && [ -n "${DST_HOST}" ]; then
        # 원격↔원격 (예: rllab에서 yonsei scratch↔scratch2 지시)
        # → 같은 호스트일 때만 그 호스트에 들어가서 로컬 rsync 실행
        [ "${SRC_HOST}" = "${DST_HOST}" ] || die "서로 다른 원격 호스트 간 직접 전송은 지원하지 않습니다."
        ssh -t "${SRC_HOST}" "mkdir -p $(remote_quote "${dst_abs}") && rsync -avh --progress $(remote_quote "${src_abs}")/ $(remote_quote "${dst_abs}")/"
        return
    fi

    if [ -z "${DST_HOST}" ]; then
        mkdir -p "${dst_abs}"
    else
        ssh "${DST_HOST}" "mkdir -p $(remote_quote "${dst_abs}")"
    fi

    # 로컬↔로컬은 압축(-z) 불필요
    local rsync_opts=(-avh --progress)
    if [ -n "${SRC_HOST}" ] || [ -n "${DST_HOST}" ]; then
        rsync_opts=(-avzh --progress)
    fi

    rsync "${rsync_opts[@]}" "${src}/" "${dst}/"
}

echo ""
echo "SBD sync_server  (mode: ${SYNC_MODE})"
if [ "${SYNC_MODE}" = "yonsei" ]; then
    echo "  * yonsei→rllab 방향 접속이 불가능해서 내부(scratch↔scratch2) 전용입니다."
    echo "  * rllab과 주고받으려면 rllab 쪽에서 이 스크립트를 실행하세요."
fi

EP_LABELS=()
for i in "${!EP_NAMES[@]}"; do
    EP_LABELS+=("$(ep_label "$i")")
done

print_menu "FROM (보내는 쪽) 선택" "${EP_LABELS[@]}"
FROM_IDX="$(choose_index "${#EP_NAMES[@]}")" || exit 0

print_menu "TO (받는 쪽) 선택" "${EP_LABELS[@]}"
TO_IDX="$(choose_index "${#EP_NAMES[@]}")" || exit 0

[ "${FROM_IDX}" != "${TO_IDX}" ] || die "FROM과 TO가 같습니다."

SRC_HOST="${EP_HOSTS[$FROM_IDX]}"
SRC_BASE="${EP_BASES[$FROM_IDX]}"
DST_HOST="${EP_HOSTS[$TO_IDX]}"
DST_BASE="${EP_BASES[$TO_IDX]}"

if [ -n "${SRC_HOST}" ]; then
    SRC_LABEL="${SRC_HOST}:${SRC_BASE}"
else
    SRC_LABEL="${SRC_BASE}"
fi

echo ""
echo "  from: ${EP_LABELS[$FROM_IDX]}"
echo "  to  : ${EP_LABELS[$TO_IDX]}"

CHOSEN_REL=""
choose_folder || exit 0
sync_folder "${CHOSEN_REL}"

echo ""
echo "완료."
