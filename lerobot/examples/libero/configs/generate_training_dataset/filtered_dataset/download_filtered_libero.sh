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
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/filtered_dataset_config.py" --shell)"

HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
[ -n "${HF_BIN}" ] || { echo "hf / huggingface-cli not found in PATH" >&2; exit 1; }

STAGING="${FILTERED_ROOT}/_v21"
mkdir -p "${STAGING}"

for pair in ${FILTERED_SUITES}; do
  name="${pair%%=*}"; repo="${pair#*=}"
  if [ -n "${FILTERED_ONLY:-}" ] && ! grep -qw "${name}" <<<"${FILTERED_ONLY}"; then
    continue
  fi
  dest="${STAGING}/${name}"
  if [ -f "${dest}/meta/info.json" ]; then
    echo "[skip] ${name}: already downloaded (${dest})"
    continue
  fi
  echo "[download] ${repo} -> ${dest}"
  "${HF_BIN}" download "${repo}" --repo-type dataset --local-dir "${dest}"
done
echo "DONE downloads → ${STAGING}"
