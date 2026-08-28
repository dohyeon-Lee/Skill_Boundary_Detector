#!/usr/bin/env bash
# Submit unified PT/FT skill predictor + FSQ terminator training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${SKILL_AUX_TRAIN_CONFIG:-${TERMINATOR_TRAIN_CONFIG:-${SCRIPT_DIR}/auxiliary_train_config.yaml}}"
if [ "$#" -ne 0 ]; then
  echo "Usage: $0 (configure mode and train targets in auxiliary_train_config.yaml)" >&2
  exit 2
fi

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do
  CONFIG_LIB="$(dirname "${CONFIG_LIB}")"
done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_train_config.py" \
    --config "${CONFIG_PATH}" --shell
)"; then
  echo "Auxiliary configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"
: "${SKILLVLA_DATASET_DIR:?Bootstrap did not export SKILLVLA_DATASET_DIR}"
if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing SkillVLA dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit auxiliary-only training"
echo "  init    : ${INITIALIZATION_MODE}"
echo "  targets : ${TRAINING_MODE}"
echo "  run     : ${RUN_NAME}"
echo "  dataset : ${SKILLVLA_DATASET_DIR}"
echo "  output  : ${OUTPUT_DIR}"

SKILL_AUX_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
