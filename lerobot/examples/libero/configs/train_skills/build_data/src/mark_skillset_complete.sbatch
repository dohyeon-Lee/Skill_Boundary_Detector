#!/usr/bin/env bash
# Inputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
# Reference model:
#   none
# Outputs:
#   completion   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/.complete
#
# Mark skillset generation complete after the array succeeds.
#SBATCH --job-name=mark_skillset
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../build_data_config.yaml}

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset skills dir missing: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

EFFECTIVE_GLOBAL_THRESHOLD_PATH="${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-${SKILLSET_GLOBAL_THRESHOLD_PATH}}"
if [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ] && [ ! -f "${EFFECTIVE_GLOBAL_THRESHOLD_PATH}" ]; then
  echo "Global threshold file missing: ${EFFECTIVE_GLOBAL_THRESHOLD_PATH}" >&2
  exit 1
fi

N_SKILLS=$(find "${SKILLSET_DIR}/skills" -name '*.npz' | wc -l)
if [ "${N_SKILLS}" -eq 0 ]; then
  echo "No skill npz files found under ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

mkdir -p "${SKILLSET_DIR}"
{
  echo "dataset=${TARGET_DATASET}"
  echo "skills=${N_SKILLS}"
  echo "boundary_threshold_mode=${SKILLSET_BOUNDARY_THRESHOLD_MODE}"
  [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ] && \
    echo "global_threshold_path=${EFFECTIVE_GLOBAL_THRESHOLD_PATH}"
  [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ] && \
    echo "global_threshold_source=${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
  echo "completed_at=$(date -Iseconds)"
} > "${SKILLSET_DONE_PATH}"

echo "Skillset complete: ${SKILLSET_DONE_PATH} (${N_SKILLS} skills)"
