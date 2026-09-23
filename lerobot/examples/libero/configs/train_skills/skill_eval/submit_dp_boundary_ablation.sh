#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${DP_BOUNDARY_ABLATION_CONFIG:-${SCRIPT_DIR}/dp_boundary_ablation_config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

_lib="$(dirname "${CONFIG}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
CONFIG="$(snapshot_config "${CONFIG}")"

if ! RESOLVED="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/dp_boundary_ablation_config.py" \
    --config "${CONFIG}" --shell
)"; then
  echo "DP boundary ablation bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"

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
[ -n "${DP_ABLATION_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DP_ABLATION_NODELIST}")
[ -n "${DP_ABLATION_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${DP_ABLATION_EXCLUDE_NODES}")

echo "Submit one-model DP boundary ablation"
echo "  skillset : ${DP_ABLATION_SKILLSET_DIR}"
echo "  output   : ${DP_ABLATION_OUTPUT_DIR}/index.html"
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/dp_boundary_ablation.sbatch"
