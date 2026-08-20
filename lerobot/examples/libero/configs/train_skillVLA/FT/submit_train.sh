#!/usr/bin/env bash
# Resolve and submit Stage-2 fine-tuning from a complete Stage-2 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FT_TRAIN_CONFIG:-${SCRIPT_DIR}/ft_train_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
mkdir -p "${SCRIPT_DIR}/logs"
FT_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/ft_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_train_config.py" \
  --config "${CONFIG_PATH}" --shell > "${FT_ENV_SNAPSHOT}"
source "${FT_ENV_SNAPSHOT}"

SBATCH_ARGS=(
  --job-name="stage2_ft_${STAGE2_MODE}"
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
[ -z "${TRAIN_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
[ -z "${TRAIN_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")

echo "Submit Stage-2 FT ${STAGE2_MODE}"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  parent   : ${STAGE2_CHECKPOINT_PATH}"
echo "  output   : ${PT_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
FT_TRAIN_DIR="${SCRIPT_DIR}" FT_TRAIN_CONFIG="${CONFIG_PATH}" FT_ENV_SNAPSHOT="${FT_ENV_SNAPSHOT}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
