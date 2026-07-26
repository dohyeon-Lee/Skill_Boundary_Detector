#!/usr/bin/env bash
# Download LangGap LeRobot v3.0 datasets (YC11Hou/langgap_*) into the staging area:
#   {project_root}/{langgap_root}/_hf/{name}
# Conversion/orientation-check/stats happen in build_langgap_dataset.sh (which calls this first).
#
# Usage:
#   ./download_langgap.sh                              # default sets from the yaml
#   LANGGAP_ONLY="langgap_6_smoke" ./download_langgap.sh   # subset (incl. extra_sets)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/langgap_dataset_config.py" --shell)"

HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
[ -n "${HF_BIN}" ] || { echo "hf / huggingface-cli not found in PATH" >&2; exit 1; }

# Robust transfer (검증된 filtered_dataset 패턴): 순정 파이썬 백엔드는 read timeout이 없어
# 죽은 소켓에 매달린다 → hf_transfer(rust) + read timeout이 끊고 자동 재시도.
if "${BOOTSTRAP_PYTHON}" -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
DL_RETRIES="${DL_RETRIES:-5}"

STAGING="${LANGGAP_ROOT}/_hf"
mkdir -p "${STAGING}"

# LANGGAP_ONLY 미지정 시 default_sets만; 지정 시 langgap_sets+extra_sets 전체에서 필터.
TARGETS="${LANGGAP_ONLY:-${DEFAULT_SETS}}"

for pair in ${LANGGAP_SETS}; do
  name="${pair%%=*}"; repo="${pair#*=}"
  grep -qw "${name}" <<<"${TARGETS}" || continue
  dest="${STAGING}/${name}"
  # Complete ONLY IF meta/info.json AND data/ AND videos/ exist, with no hf staging leftovers
  # (*.incomplete / *.lock). meta-only 잔해(중단된 다운로드)가 완료로 오판되는 것을 막는다.
  if [ -f "${dest}/meta/info.json" ] && \
     [ -n "$(find "${dest}/data" -name '*.parquet' 2>/dev/null | head -1)" ] && \
     [ -n "$(find "${dest}/videos" -name '*.mp4' 2>/dev/null | head -1)" ] && \
     [ -z "$(find "${dest}" -name '*.incomplete' -o -name '*.lock' 2>/dev/null | head -1)" ]; then
    echo "[skip] ${name}: already downloaded (${dest})"
    continue
  fi
  echo "== download ${name} <- ${repo} =="
  mkdir -p "${dest}"
  ok=""
  for i in $(seq 1 "${DL_RETRIES}"); do
    if "${HF_BIN}" download "${repo}" --repo-type dataset --local-dir "${dest}"; then
      ok=1; break
    fi
    echo "  retry ${i}/${DL_RETRIES} ..."
    sleep 5
  done
  [ -n "${ok}" ] || { echo "[error] ${name}: download failed after ${DL_RETRIES} retries" >&2; exit 1; }
  [ -f "${dest}/meta/info.json" ] || { echo "[error] ${name}: meta/info.json missing after download" >&2; exit 1; }
  echo "== ${name}: downloaded -> ${dest} =="
done

echo "DOWNLOAD DONE -> ${STAGING}"
