#!/usr/bin/env bash
# Download the FILTERED (OpenVLA no_noops) LIBERO suites from IPEC-COMMUNITY in their
# original v2.1 format, into the staging area:
#   {project_root}/{filtered_dataset_root}/_v21/{name}
# Conversion/remap/stats happen in build_filtered_dataset.sh (which calls this first).
#
# Usage:
#   ./download_filtered_libero.sh                 # all suites from the yaml
#   FILTERED_ONLY="libero_spatial_full_full" ./download_filtered_libero.sh   # subset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/filtered_dataset_config.py" --shell)"

HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
[ -n "${HF_BIN}" ] || { echo "hf / huggingface-cli not found in PATH" >&2; exit 1; }

# Robust transfer: the pure-python backend has no read timeout, so a stalled connection hangs forever
# on a single file (observed: 321 kB/s then dead, *.incomplete stuck at 0B). hf_transfer (rust) plus a
# read timeout drops the hung socket and retries automatically. Enable hf_transfer only if importable.
if "${BOOTSTRAP_PYTHON}" -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
DL_RETRIES="${DL_RETRIES:-5}"

STAGING="${FILTERED_ROOT}/_v21"
mkdir -p "${STAGING}"

for pair in ${FILTERED_SUITES}; do
  name="${pair%%=*}"; repo="${pair#*=}"
  if [ -n "${FILTERED_ONLY:-}" ] && ! grep -qw "${name}" <<<"${FILTERED_ONLY}"; then
    continue
  fi
  dest="${STAGING}/${name}"
  # Complete ONLY IF meta/info.json exists AND no hf staging leftovers (*.incomplete / *.lock, which hf
  # leaves on a partial/stalled run and removes on success). info.json alone is downloaded early, so a
  # stalled download would falsely look "done". A leftover → fall through and re-run hf (idempotent:
  # resumes, fetching the missing/incomplete files; already-complete files are skipped via etag).
  if [ -f "${dest}/meta/info.json" ] \
     && ! find "${dest}" \( -name '*.incomplete' -o -name '*.lock' \) 2>/dev/null | grep -q .; then
    echo "[skip] ${name}: already downloaded (${dest})"
    continue
  fi
  echo "[download] ${repo} -> ${dest}"
  # Retry loop: each hf download resumes from the *.incomplete files, so a stalled attempt costs only the
  # in-flight file. hf_transfer + the read timeout above make a single stall self-recover, but keep the
  # loop for the pathological case where the process wedges past the timeout.
  ok=0
  for attempt in $(seq 1 "${DL_RETRIES}"); do
    if "${HF_BIN}" download "${repo}" --repo-type dataset --local-dir "${dest}"; then
      : # hf returned 0; still verify leftovers below (it can exit 0 with a resumable partial)
    else
      echo "[retry] ${name}: hf download exited non-zero (attempt ${attempt}/${DL_RETRIES})" >&2
    fi
    if ! find "${dest}" \( -name '*.incomplete' -o -name '*.lock' \) 2>/dev/null | grep -q .; then
      ok=1; break
    fi
    echo "[retry] ${name}: incomplete files remain, resuming (attempt ${attempt}/${DL_RETRIES})" >&2
  done
  # Verify no leftovers survived (a persistent stall keeps a *.incomplete) — fail loud instead of a
  # silent partial that later builds inherit.
  if [ "${ok}" -ne 1 ]; then
    echo "[ERROR] ${name}: still incomplete after ${DL_RETRIES} attempts (network stall?). Re-run to resume." >&2
    exit 1
  fi
done
echo "DONE downloads → ${STAGING}"
