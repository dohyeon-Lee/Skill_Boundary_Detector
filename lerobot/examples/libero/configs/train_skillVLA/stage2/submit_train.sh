#!/usr/bin/env bash
# Resolve and submit selectable likelihood or DSBC Stage-2 training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE2_TRAIN_CONFIG:-${SCRIPT_DIR}/stage2_train_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
mkdir -p "${SCRIPT_DIR}/logs"
STAGE2_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/stage2_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_train_config.py" \
  --config "${CONFIG_PATH}" --shell > "${STAGE2_ENV_SNAPSHOT}"
source "${STAGE2_ENV_SNAPSHOT}"

SBATCH_ARGS=(
  --job-name="stage2_${STAGE2_MODE}"
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}"); fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}"); fi

echo "Submit Stage-2 ${STAGE2_MODE}"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  prior    : ${STAGE1_CHECKPOINT_PATH}"
echo "  predictor: ${PREDICTOR_CHECKPOINT_PATH}"
echo "  output   : ${PT_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
STAGE2_TRAIN_CONFIG="${CONFIG_PATH}" STAGE2_ENV_SNAPSHOT="${STAGE2_ENV_SNAPSHOT}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
