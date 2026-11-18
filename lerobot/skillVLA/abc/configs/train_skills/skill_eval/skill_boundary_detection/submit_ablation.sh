#!/usr/bin/env bash
# Expensive optional ablation: recompute DP probes for full-vs-no-gripper comparison.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ABC_SBD_EVAL_CONFIG:-${SCRIPT_DIR}/config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

_lib="$(dirname "${CONFIG}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG="$(snapshot_config "${CONFIG}")"

if ! RESOLVED="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --shell)"; then
  echo "ABC SBD ablation config failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"

cd "${SCRIPT_DIR}"
mkdir -p logs outputs
SBATCH_ARGS=(
  --job-name=ABC_SBD_ablation
  --partition="${DP_ABLATION_PARTITION}"
  --qos="${DP_ABLATION_QOS}"
  --gres="${DP_ABLATION_GRES}"
  --cpus-per-task="${DP_ABLATION_CPUS}"
  --mem="${DP_ABLATION_MEM}"
  --time="${DP_ABLATION_TIME}"
  --export="ALL,ABC_SBD_EVAL_DIR=${SCRIPT_DIR},ABC_SBD_EVAL_CONFIG=${CONFIG}"
)
[ -n "${DP_ABLATION_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DP_ABLATION_NODELIST}")
[ -n "${DP_ABLATION_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${DP_ABLATION_EXCLUDE_NODES}")

echo "Submit optional ABC SBD full-vs-no-gripper ablation"
echo "  skillset: ${DP_ABLATION_SKILLSET_DIR}"
echo "  report  : ${DP_ABLATION_OUTPUT_DIR}/index.html"
submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/run.sbatch"
