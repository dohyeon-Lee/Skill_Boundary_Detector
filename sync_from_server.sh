#!/usr/bin/env bash
# Fetch ignored dataset/model/output folders from another server into this SBD workspace.
#
# Usage:
#   ./sync_from_server.sh
#   REMOTE=user@host REMOTE_BASE=/path/to/SBD ./sync_from_server.sh
#
# Flow:
#   1. Pick one top-level remote SBD folder from the large-data/output .gitignore set.
#   2. Fetch that folder as-is, or keep drilling down into subfolders.
#   3. The chosen folder is synced to the same relative path under this local SBD.

set -euo pipefail

REMOTE="${REMOTE:-mdorazi@165.132.142.207}"
REMOTE_BASE="${REMOTE_BASE:-/scratch/mdorazi/Skill_Boundary_Detector}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="${LOCAL_BASE:-${SCRIPT_DIR}}"

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

remote_find_dirs() {
    local rel="${1:-}"
    local base="${REMOTE_BASE}"
    if [ -n "${rel}" ]; then
        base="${REMOTE_BASE}/${rel}"
    fi
    local quoted
    quoted="$(remote_quote "${base}")"
    ssh "${REMOTE}" "find ${quoted} -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort"
}

list_top_dirs() {
    local base quoted patterns
    base="${REMOTE_BASE}"
    quoted="$(remote_quote "${base}")"
    patterns="$(printf '%q ' "${SYNC_ROOT_PATTERNS[@]}")"

    ssh "${REMOTE}" "cd ${quoted} 2>/dev/null && for pattern in ${patterns}; do for d in \$pattern; do [ -d \"\$d\" ] && printf '%s\n' \"\$d\"; done; done | awk '!seen[\$0]++'"
}

choose_folder() {
    local top_dirs=()
    mapfile -t top_dirs < <(list_top_dirs)
    [ "${#top_dirs[@]}" -gt 0 ] || die "No output/dataset folders found under ${REMOTE}:${REMOTE_BASE}"

    print_menu "가져올 원격 최상위 폴더 선택 (${REMOTE}:${REMOTE_BASE})" "${top_dirs[@]}"
    local idx
    idx="$(choose_index "${#top_dirs[@]}")" || return 1

    local rel="${top_dirs[$idx]}"
    while true; do
        local remote_path="${REMOTE_BASE}/${rel}"
        if ! ssh "${REMOTE}" "test -d $(remote_quote "${remote_path}")"; then
            die "Remote folder not found: ${REMOTE}:${remote_path}"
        fi

        echo ""
        echo "현재 선택: ${rel}"
        echo "  [1] 이 폴더 전체 가져오기"
        echo "  [2] 하위폴더 고르기"
        echo "  [q] cancel"
        echo ""

        local action
        read -r -p "선택: " action
        case "${action}" in
            1) CHOSEN_REL="${rel}"; return 0 ;;
            2)
                local children=()
                mapfile -t children < <(remote_find_dirs "${rel}")
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
    local src="${REMOTE}:${REMOTE_BASE}/${rel}"
    local dst="${LOCAL_BASE}/${rel}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Sync from server"
    echo "  remote: ${src}"
    echo "  local : ${dst}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    mkdir -p "${dst}"
    rsync -avzh --progress "${src}/" "${dst}/"
}

echo ""
echo "SBD sync_from_server"
echo "  remote     : ${REMOTE}"
echo "  remote base: ${REMOTE_BASE}"
echo "  local base : ${LOCAL_BASE}"

CHOSEN_REL=""
choose_folder || exit 0
sync_folder "${CHOSEN_REL}"

echo ""
echo "완료."
