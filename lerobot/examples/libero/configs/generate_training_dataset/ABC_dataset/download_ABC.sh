#!/usr/bin/env bash
# Download ABC-130k mcap SUBSETs (per ABC_dataset_config.yaml abc_subsets) into the
# staging area: {project_root}/{abc_dataset_root}/_mcap/{name}/data/{split}/...
# 엔진 = {abcdl_repo}/download/src/download_abc.py (그룹/태스크/에피소드 선택형 다운로더;
# 논문 7개 primitive 카테고리 taxonomy 내장). Conversion happens in build_ABC_dataset.sh.
#
# NOTE: XDOF/ABC-130k는 gated — HF 페이지에서 라이선스 수락 + `huggingface-cli login` 선행.
#
# 자동 hang 복구(WATCHDOG): HF 소켓이 죽어 전송이 멈추면(read timeout 부재로 무한 대기)
# 스테이징 용량 증가가 멈춘다. 이를 감지해 "내가 띄운 다운로드 프로세스 그룹만" 종료하고
# 재시작한다(hf 다운로드는 idempotent — 완료분 스킵, .incomplete 이어받기). --dry-run /
# --list-tasks 에는 미적용.
#   STALL_TIMEOUT   : 다운로드 시작 후 용량 무증가 이 초를 넘기면 hang으로 판정 (기본 180)
#   LISTING_GRACE   : 첫 바이트 전(에피소드 listing/접속) 허용 무증가 시간 (기본 600)
#   CHECK_INTERVAL  : 폴링 주기 초 (기본 20)
#   DL_MAX_RETRIES  : 재시작 횟수 상한 (기본 1000 = 사실상 무한, 백스톱)
#   DL_RETRY_BACKOFF: 재시작 전 대기 초 — 소켓 정리 여유 (기본 10)
#   WATCHDOG=0      : 워치독 끄고 한 번만 실행
#
# Usage:
#   ./download_ABC.sh --list-tasks           # 태스크 폴더명 전체 (--counts 로 에피소드 수까지)
#   ./download_ABC.sh --dry-run              # 계획만 출력 (다운로드 X)  (DRY_RUN=1 도 동일)
#   ./download_ABC.sh                        # all subsets from the yaml (+ 자동 hang 복구)
#   ABC_ONLY="abc_toy" ./download_ABC.sh     # subset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

# ── 대용량 파일 다운로드 안정화 (original_dataset의 검증된 레시피) ──────────────
# XDOF/ABC-130k의 mcap(230~320MB)은 HF의 Xet(CAS 청크) 경로로 서빙되는데 이게 자주 스톨한다
# (.incomplete 파일명의 'r2MXud…=' 접두사 = Xet 토큰으로 확인). Xet을 끄면 재래식 LFS 전송으로
# 되돌아가고, read/etag 타임아웃이 죽은 소켓을 끊어 재시도가 걸린다. original_dataset에서 검증됨.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

# hf가 다운로드 중이면 .incomplete/.lock 을 남기고, 완료 시 지운다(filtered_dataset의 완료 기준).
# 이게 "진짜 hang(받을 게 남아 무증가)"과 "완료(받을 게 없어 무증가)"를 가르는 핵심 신호다.
_has_pending() {
  find "$1" \( -name '*.incomplete' -o -name '*.lock' \) 2>/dev/null | grep -q .
}

# Run the download engine under a stall-watchdog, restarting on hang until it exits cleanly.
# watch_dir 의 총 바이트가 늘면 진행 중. 무증가 시: .incomplete 있으면 hang→재시작, 없으면 완료로 판정.
run_with_watchdog() {
  local watch_dir="$1"; shift
  local stall="${STALL_TIMEOUT:-180}" interval="${CHECK_INTERVAL:-20}"
  local listing_grace="${LISTING_GRACE:-600}" max_retries="${DL_MAX_RETRIES:-1000}"
  local backoff="${DL_RETRY_BACKOFF:-10}" complete_confirm="${COMPLETE_CONFIRM:-45}"
  mkdir -p "${watch_dir}"

  # 이미 다 받았으면(데이터 존재 + pending 없음) 엔진을 아예 띄우지 않고 즉시 통과 —
  # 완료된 데이터셋에 대한 불필요한 재검증/재시작 루프를 원천 차단.
  if ! _has_pending "${watch_dir}" \
     && [ -n "$(find "${watch_dir}" -name 'episode.mcap' 2>/dev/null | head -1)" ]; then
    echo "[watchdog] ✅ 이미 완료 (.incomplete/.lock 없음) → 다운로드 스킵"
    return 0
  fi

  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    echo "[watchdog] 다운로드 시도 #${attempt} (stall=${stall}s, listing-grace=${listing_grace}s)"

    # bash job-control(monitor)로 실행 → 이 백그라운드 잡이 자기 프로세스 그룹을 가짐
    # (PGID==child PID). 재시작 시 이 그룹만 kill 하므로 다른 사람 프로세스는 절대 안 건드림.
    # (setsid & 는 런처가 즉시 detach돼 $!가 워크로드를 못 잡으므로 쓰지 않음.)
    set -m
    "$@" &
    local pid=$! pgid=$!
    set +m

    local last_size=0 last_progress=${SECONDS} started=0 outcome=""
    while :; do
      if ! kill -0 "${pid}" 2>/dev/null; then          # 프로세스 종료됨
        local rc=0; wait "${pid}" 2>/dev/null || rc=$?
        outcome="exit:${rc}"; break
      fi
      sleep "${interval}"
      local size; size=$(du -sb "${watch_dir}" 2>/dev/null | cut -f1)
      [ -z "${size}" ] && continue                      # du 실패 → 판단 보류
      if [ "${size}" -gt "${last_size}" ]; then
        last_size=${size}; last_progress=${SECONDS}; started=1
      elif ! _has_pending "${watch_dir}"; then
        # 용량 무증가 + 받을 것 없음(.incomplete/.lock 0) = 완료. 엔진이 최종 검증으로
        # 무증가일 뿐이니 hang이 아니다. 짧게 확인(complete_confirm) 후 완료로 판정.
        local idle=$((SECONDS - last_progress))
        if [ "${idle}" -ge "${complete_confirm}" ]; then
          echo "[watchdog] 완료 감지: .incomplete/.lock 없음 + ${idle}s 무증가 → 다운로드 완료"
          kill -TERM -- "-${pgid}" 2>/dev/null || true   # 검증만 돌던 엔진 정리
          sleep 3; kill -KILL -- "-${pgid}" 2>/dev/null || true
          wait "${pid}" 2>/dev/null || true
          outcome="complete"; break
        fi
      else
        # 용량 무증가 + 받을 것 남음(.incomplete 존재) = 진짜 hang → 그룹 kill 후 재시작.
        local idle=$((SECONDS - last_progress))
        local limit=${stall}; [ "${started}" -eq 0 ] && limit=${listing_grace}
        if [ "${idle}" -ge "${limit}" ]; then
          echo "[watchdog] STALL: ${idle}s 무증가 + 미완료 파일 존재 ($(numfmt --to=iec ${last_size} 2>/dev/null || echo ${last_size})) → 그룹 ${pgid} 종료 후 재시작"
          kill -TERM -- "-${pgid}" 2>/dev/null || true
          sleep 5
          kill -KILL -- "-${pgid}" 2>/dev/null || true
          wait "${pid}" 2>/dev/null || true
          outcome="stalled"; break
        fi
      fi
    done

    if [ "${outcome}" = "exit:0" ] || [ "${outcome}" = "complete" ]; then
      echo "[watchdog] ✅ 다운로드 완료"
      return 0
    fi
    if [ "${attempt}" -ge "${max_retries}" ]; then
      echo "[watchdog] ${attempt}회 시도 후 중단 (마지막: ${outcome})" >&2
      return 1
    fi
    echo "[watchdog] ${outcome} → ${backoff}s 후 재시작 (완료분 스킵, 이어받기)…"
    sleep "${backoff}"
  done
}

echo "== ABC subset download (repo: ${ABC_HF_REPO}, engine: ${ABCDL_REPO}/download) =="

# --dry-run / --list-tasks (또는 DRY_RUN) 은 파일을 안 쓰므로 워치독 미적용 — 바로 실행.
_real=1
{ [ -n "${DRY_RUN:-}" ] && [ "${DRY_RUN}" != "0" ]; } && _real=0
for _a in "$@"; do case "${_a}" in --list-tasks|--dry-run) _real=0 ;; esac; done
export ABC_ONLY="${ABC_ONLY:-}" DRY_RUN="${DRY_RUN:-}"

if [ "${_real}" -eq 1 ] && [ "${WATCHDOG:-1}" != "0" ]; then
  run_with_watchdog "${ABC_ROOT}/_mcap" \
    "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/download_abc_subset.py" "$@"
else
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/download_abc_subset.py" "$@"
fi
