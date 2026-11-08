#!/usr/bin/env bash
# Parallel local filesystem copy used by Transfer Desk.
set -euo pipefail

parallel_local_copy() {
    local src_abs="$1" dst_abs="$2" jobs="$3"
    mkdir -p "${dst_abs}"
    # 1) 디렉토리 트리(빈 폴더 포함)를 먼저 만들어 둔다 → 병렬 rsync는 파일만 채우고
    #    같은 부모 폴더를 동시에 mkdir 하는 경쟁을 피한다.
    rsync -a -f'+ */' -f'- *' "${src_abs}/" "${dst_abs}/"

    # 전체 진행률 표시용 총 항목 수(파일+폴더+링크). 폴더는 위에서 이미 생성됨.
    local total
    total=$(cd "${src_abs}" && find . 2>/dev/null | wc -l)

    # 2) 파일/심링크 목록을 null 구분으로 뽑아 jobs개 병렬 전송(백그라운드).
    #    각 rsync도 src로 cd 후 실행해야 './sub/file'가 맞게 풀리고,
    #    rsync -aR 의 '.' 앵커로 디렉토리 구조가 그대로 재현된다.
    ( cd "${src_abs}" && find . \( -type f -o -type l \) -print0 ) \
        | xargs -0 -r -P "${jobs}" -n 256 \
            sh -c 'src=$1; dst=$2; shift 2; cd "$src" && rsync -aR "$@" "$dst"' \
               sh "${src_abs}" "${dst_abs}/" &
    local copy_pid=$!

    # 3) 진행률 모니터: dst 항목 수를 세어 한 줄로 갱신 표시.
    local cnt pct
    while kill -0 "${copy_pid}" 2>/dev/null; do
        cnt=$(find "${dst_abs}" 2>/dev/null | wc -l)
        pct=$(( total > 0 ? cnt * 100 / total : 100 ))
        printf '\r  진행: %d/%d (%d%%)   ' "${cnt}" "${total}" "${pct}"
        sleep 3
    done
    local rc=0
    wait "${copy_pid}" || rc=$?
    cnt=$(find "${dst_abs}" 2>/dev/null | wc -l)
    printf '\r  진행: %d/%d (100%%)   완료\n' "${cnt}" "${total}"
    return "${rc}"
}
