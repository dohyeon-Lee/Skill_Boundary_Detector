#!/usr/bin/env bash
# Submit ABC Diffusion Policy training using dp_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ABC_DP_CONFIG:-${SCRIPT_DIR}/dp_config.yaml}"

# Freeze the yaml so a queued job cannot observe later edits.
_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
if ! CONFIG_EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/train_dp_config.py" \
  --config "${CONFIG_PATH}" --shell)"; then
  echo "ABC DP configuration failed; no job was submitted." >&2
  exit 1
fi
eval "${CONFIG_EXPORTS}"

SBATCH_ARGS=(
  --partition="${DP_PARTITION}"
  --qos="${DP_QOS}"
  --gres="${DP_GRES}"
  --cpus-per-task="${DP_CPUS_PER_TASK}"
  --mem="${DP_MEM}"
  --time="${DP_TIME}"
)
[ -n "${DP_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DP_NODELIST}")
[ -n "${DP_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${DP_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit ABC DP"
echo "  dataset : ${DP_DATASET_DIR}"
echo "  tasks   : ${DP_TASKS_LABEL} -> ${DP_RESOLVED_TASKS_LABEL}"
echo "  input   : 14D absolute YAM joints (state only)"
echo "  horizon : obs=${DP_N_OBS_STEPS}, future=${DP_FUTURE_ACTION_HORIZON} @ 30 Hz"
echo "  output  : ${DP_OUTPUT_DIR}"
echo "  save    : every ${DP_SAVE_FREQ} steps through ${DP_STEPS}"
echo "  slurm   : partition=${DP_PARTITION} qos=${DP_QOS} gres=${DP_GRES}"

if [ "${TRAIN_DP}" != true ]; then
  echo "Skip ABC DP because train_DP=false"
  exit 0
fi
if [ -d "${DP_FINAL_CHECKPOINT}" ]; then
  echo "Skip ABC DP because the 100k checkpoint already exists: ${DP_FINAL_CHECKPOINT}"
  exit 0
fi

ABC_DP_CONFIG="${CONFIG_PATH}" submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/train_dp.sbatch"
