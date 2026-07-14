#!/usr/bin/env bash
# check_usage.sh — 서버 / 스토리지 볼륨을 자동 인식하는 디스크 사용량 요약
#
# 여러 서버(rllab, yonsei ...)와 서버 안 여러 스토리지 볼륨
# (/data1,2,3 · /scratch,2 · $HOME ...)을 파일 하나로 자동 인식해서 요약한다.
# 기존 check_disk_usage.sh / check_dohyeon_disk_usage.sh /
# check_mdorazi_disk_usage.sh 를 대체한다.
#
# 자동 인식 방식:
#   1) 후보 스토리지 베이스(BASES) 중 실제 존재하는 것만 추림
#   2) 각 베이스에서 사용자 디렉토리 <base>/<user> (USERS 목록)를 찾음 → "루트"
#   3) $HOME 도 루트에 포함
#   4) 루트마다  df(마운트 여유공간) + 상위 폴더 용량 순위 를 출력
#      → 서버가 바뀌거나 새 볼륨(/data3 등)이 생겨도 손 안 대고 자동으로 잡힌다.
#
# 사용법:
#   ./check_usage.sh                     # 자동 인식, 상위 폴더 순위
#   DEPTH=2 ./check_usage.sh             # + 한 단계 하위폴더까지 상세 (트리)
#   DEPTH=3 ./check_usage.sh             # + 두 단계까지
#   MIN_SIZE=1G ./check_usage.sh         # 1G 미만 숨기기
#   INCLUDE_HIDDEN=0 ./check_usage.sh    # 숨김 폴더(.으로 시작) 제외
#   ROOTS="/data2/dohyeon/SBD" ./check_usage.sh   # 루트 직접 지정(자동인식 생략)
#   USERS="dohyeon mdorazi" ./check_usage.sh      # 탐색할 사용자명 목록
#   BASES="/data1 /data2 /scratch" ./check_usage.sh
#
# 환경변수:
#   USERS           탐색 사용자명 목록 (기본: "$USER", 못 찾으면 dohyeon mdorazi 도 시도)
#   BASES           후보 스토리지 베이스 (기본 아래 DEFAULT_BASES)
#   ROOTS           루트 직접 지정 시 자동 인식 생략 (공백 구분)
#   DEPTH           하위 폴더 상세 깊이. 1=상위폴더 순위만, 2=한단계 더, ... (기본 1)
#   MIN_SIZE        이 값보다 작은 항목 숨김, e.g. 1G / 500M (기본 0 = 전체)
#   INCLUDE_HIDDEN  숨김 폴더 포함: 1 포함 / 0 제외 (기본 1)

set -euo pipefail

DEFAULT_BASES="/data1 /data2 /data3 /data4 /scratch /scratch2 /scratch3"
DEFAULT_USERS="${USER:-} dohyeon mdorazi"

DEPTH="${DEPTH:-1}"
MIN_SIZE="${MIN_SIZE:-0}"
INCLUDE_HIDDEN="${INCLUDE_HIDDEN:-1}"

read -r -a BASES <<< "${BASES:-$DEFAULT_BASES}"
read -r -a USERS <<< "${USERS:-$DEFAULT_USERS}"

(( DEPTH >= 1 )) || { echo "DEPTH 는 1 이상이어야 합니다." >&2; exit 1; }

# MIN_SIZE → 바이트
bytes_min=0
if [[ "$MIN_SIZE" != "0" ]]; then
    bytes_min=$(numfmt --from=iec "$MIN_SIZE" 2>/dev/null || echo 0)
fi

# 사람이 읽는 크기로 변환
human() { numfmt --to=iec "$1" 2>/dev/null || echo "${1}B"; }

# ── 서버 이름 추정 (표시용) ───────────────────────────────────────────────────
HN="$(hostname)"
case "$HN" in
    yonsei*)          SERVER="yonsei" ;;
    node*|rllab*)     SERVER="rllab" ;;
    *)                SERVER="$HN" ;;
esac

# ── 루트 자동 인식: <base>/<user> 중 실제 존재하는 것 + $HOME ─────────────────
ROOTS_ARR=()
_add_root() {
    local p="$1" existing
    [ -d "$p" ] || return 0
    p="$(cd "$p" && pwd -P)"   # 심볼릭 링크 정규화 → 중복 제거 정확도↑
    for existing in ${ROOTS_ARR[@]+"${ROOTS_ARR[@]}"}; do
        [ "$existing" = "$p" ] && return 0
    done
    ROOTS_ARR+=("$p")
}

if [[ -n "${ROOTS:-}" ]]; then
    read -r -a _explicit <<< "$ROOTS"
    for r in "${_explicit[@]}"; do _add_root "$r"; done
else
    for base in "${BASES[@]}"; do
        [ -d "$base" ] || continue
        for u in "${USERS[@]}"; do
            [ -n "$u" ] && _add_root "$base/$u"
        done
    done
    _add_root "${HOME:-}"
fi

if (( ${#ROOTS_ARR[@]} == 0 )); then
    echo "인식된 스토리지 루트가 없습니다." >&2
    echo "  hostname=$HN  user=${USER:-?}" >&2
    echo "  BASES=(${BASES[*]})  USERS=(${USERS[*]})" >&2
    echo "  ROOTS 환경변수로 직접 지정할 수 있습니다." >&2
    exit 1
fi

# ── 헤더 ─────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "  디스크 사용량  |  서버: ${SERVER}  |  사용자: ${USER:-?}  |  DEPTH=${DEPTH}"
[[ "$MIN_SIZE" != "0" ]] && echo "  MIN_SIZE=${MIN_SIZE} 미만 숨김"
echo "========================================================================"
echo
echo "인식된 스토리지 루트: ${#ROOTS_ARR[@]}개  (df 여유공간)"
printf "  %-26s %8s %8s %8s %6s  %s\n" "ROOT" "SIZE" "USED" "AVAIL" "USE%" "MOUNT"
printf "  %-26s %8s %8s %8s %6s  %s\n" "----" "----" "----" "-----" "----" "-----"
for root in "${ROOTS_ARR[@]}"; do
    df -h "$root" 2>/dev/null | awk -v r="$root" 'NR==2{
        printf "  %-26s %8s %8s %8s %6s  %s\n", r, $2, $3, $4, $5, $6
    }'
done
echo

# ── 루트별 상세 ──────────────────────────────────────────────────────────────
# 각 루트를 du 한 번만 순회(--max-depth=DEPTH)해서 상위폴더 순위 + 하위 트리 출력
report_root() {
    local root="$1"
    local -a lines tops
    local -A CHILD          # "부모rel" → 자식 "bytes\trel" 줄 모음(개행 구분)
    local root_bytes=0

    echo "════════════════════════════════════════════════════════════════════════"
    echo "  ▸ ${root}/"
    echo "════════════════════════════════════════════════════════════════════════"

    mapfile -t lines < <(du -B1 --max-depth="$DEPTH" "$root" 2>/dev/null)
    if (( ${#lines[@]} == 0 )); then
        echo "    (읽을 수 없음 / 비어 있음)"; echo; return
    fi

    tops=()
    local line b p rel parent
    for line in "${lines[@]}"; do
        b="${line%%$'\t'*}"
        p="${line#*$'\t'}"
        [ "$p" = "$root" ] && { root_bytes="$b"; continue; }
        rel="${p#"$root"/}"

        # 숨김 폴더 필터 (경로 어디든 .으로 시작하면 제외)
        if [[ "$INCLUDE_HIDDEN" == "0" ]]; then
            case "$rel" in .*|*/.*) continue ;; esac
        fi
        # 최소 크기 필터
        (( bytes_min > 0 && b < bytes_min )) && continue

        if [[ "$rel" != */* ]]; then
            tops+=("$b"$'\t'"$rel")
        else
            parent="${rel%/*}"
            CHILD["$parent"]+="${b}"$'\t'"${rel}"$'\n'
        fi
    done

    echo "  [합계: $(human "$root_bytes")]"
    echo

    if (( ${#tops[@]} == 0 )); then
        if (( bytes_min > 0 )); then
            echo "    (MIN_SIZE=${MIN_SIZE} 이상 상위 폴더 없음)"
        else
            echo "    (상위 폴더 없음)"
        fi
        echo; return
    fi

    # 상위 폴더 용량 순위 (큰 것부터)
    local rank=1 sorted
    sorted="$(printf '%s\n' "${tops[@]}" | sort -t$'\t' -k1,1 -rn)"
    while IFS=$'\t' read -r b rel; do
        [ -z "$rel" ] && continue
        printf "  %2d. %8s  %s/\n" "$rank" "$(human "$b")" "$rel"
        ((rank++))
        # DEPTH>=2 이면 이 상위 폴더 아래 하위 트리를 들여쓰기해서 출력
        (( DEPTH >= 2 )) && print_children "$rel" 1
    done <<< "$sorted"
    echo
}

# CHILD 맵을 재귀적으로 따라가며 하위 폴더를 크기순·들여쓰기로 출력
print_children() {
    local parent="$1" level="$2"
    local block="${CHILD[$parent]:-}"
    [ -n "$block" ] || return 0
    local indent b rel name
    indent="$(printf '      %.0s' $(seq 1 "$level"))"
    while IFS=$'\t' read -r b rel; do
        [ -z "$rel" ] && continue
        name="${rel##*/}"
        printf "%s%8s  %s/\n" "$indent" "$(human "$b")" "$name"
        print_children "$rel" $((level + 1))
    done <<< "$(printf '%s' "$block" | sort -t$'\t' -k1,1 -rn)"
}

for root in "${ROOTS_ARR[@]}"; do
    report_root "$root"
done
