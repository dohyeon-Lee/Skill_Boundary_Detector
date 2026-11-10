#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../src/submit_job.sh"
CONFIG="${DP_BOUNDARY_ABLATION_CONFIG:-${SCRIPT_DIR}/config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ "${SBD_EVAL_HF_READY:-0}" != 1 ]; then
  PROJECT_ROOT_HINT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
  SERVER="$("${BOOTSTRAP_PYTHON}" "${PROJECT_ROOT_HINT}/lerobot/examples/libero/configs/src/global_config_loader.py" --key server)"
  HF_REQUESTED=0
  for arg in "$@"; do
    case "$arg" in --prepare-runpod|--download-only|--dry-run) HF_REQUESTED=1 ;; esac
  done
  if [ "${SERVER}" = runpod ] || [ "${HF_REQUESTED}" = 1 ]; then
    exec "${BOOTSTRAP_PYTHON}" "${PROJECT_ROOT_HINT}/tools/runpod/prepare_eval.py" --config "${CONFIG}" "$@"
  fi
fi
if [ "$#" -gt 0 ]; then
  echo "Usage: bash submit.sh [--prepare-runpod | --download-only] [--dry-run]" >&2
  exit 2
fi

_lib="$(dirname "${CONFIG}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
CONFIG="$(snapshot_config "${CONFIG}")"

if ! RESOLVED="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/config.py" \
    --config "${CONFIG}" --shell
)"; then
  echo "DP boundary ablation bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"
export EVAL_OUT_DIR="${DP_ABLATION_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
mkdir -p logs outputs
SBATCH_ARGS=(
  --job-name=DPbound
  --partition="${DP_ABLATION_PARTITION}"
  --qos="${DP_ABLATION_QOS}"
  --gres="${DP_ABLATION_GRES}"
  --cpus-per-task="${DP_ABLATION_CPUS}"
  --mem="${DP_ABLATION_MEM}"
  --time="${DP_ABLATION_TIME}"
  --export="ALL,DP_ABLATION_DIR=${SCRIPT_DIR},DP_ABLATION_CONFIG=${CONFIG}"
)
[ -n "${DP_ABLATION_DEPENDENCY:-}" ] && SBATCH_ARGS+=(--dependency="${DP_ABLATION_DEPENDENCY}")
[ -n "${DP_ABLATION_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DP_ABLATION_NODELIST}")
[ -n "${DP_ABLATION_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${DP_ABLATION_EXCLUDE_NODES}")

echo "Submit one-model DP boundary ablation"
echo "  ablation: ${DP_ABLATION_SELECTION}"
echo "  skillset : ${DP_ABLATION_SKILLSET_DIR}"
echo "  output   : ${DP_ABLATION_OUTPUT_DIR}/index.html"
bash "${_SUBMIT_JOB_SRC}/submit_job.sh" batch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/run.sbatch"
