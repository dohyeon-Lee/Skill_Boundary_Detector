#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${NEWTASK_FT_JOINT_CONFIG:-${SCRIPT_DIR}/joint_ft_config.yaml}"
if [ "$#" -ne 0 ]; then
  echo "Usage: $0" >&2
  exit 2
fi

_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then BOOTSTRAP_PYTHON=python3; fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/newtask_ft_joint_config.py" --config "${CONFIG_PATH}" --shell
)"; then
  echo "NewTask Joint configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}"); fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}"); fi

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit NewTask Joint (${ARCHITECTURE_LABEL})"
echo "  VSA      : ${VSA_CHECKPOINT_PATH}"
echo "  Predictor: ${PREDICTOR_CHECKPOINT_PATH}"
echo "  Terminator: ${TERMINATOR_CHECKPOINT_PATH:-disabled}"
echo "  run      : ${RUN_NAME}"
echo "  output   : ${OUTPUT_DIR}"
if [ "${NEWTASK_FT_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: sbatch ${SBATCH_ARGS[*]} ${SRC_DIR}/train.sbatch"
  exit 0
fi

NEWTASK_FT_JOINT_CONFIG="${CONFIG_PATH}" \
NEWTASK_FT_JOINT_RESOLVER="${SRC_DIR}/newtask_ft_joint_config.py" \
NEWTASK_FT_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}" \
  submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
