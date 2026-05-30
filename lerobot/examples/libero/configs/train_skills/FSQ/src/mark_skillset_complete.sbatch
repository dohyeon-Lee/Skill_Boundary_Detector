#!/usr/bin/env bash
# Inputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
# Reference model:
#   none
# Outputs:
#   completion   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/.complete
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
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../../train_skills_config.yaml}

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

N_SKILLS=$(find "${SKILLSET_DIR}/skills" -name '*.npz' | wc -l)
if [ "${N_SKILLS}" -eq 0 ]; then
  echo "No skill npz files found under ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

mkdir -p "${SKILLSET_DIR}"
{
  echo "dataset=${TARGET_DATASET}"
  echo "skills=${N_SKILLS}"
  echo "completed_at=$(date -Iseconds)"
} > "${SKILLSET_DONE_PATH}"

echo "Skillset complete: ${SKILLSET_DONE_PATH} (${N_SKILLS} skills)"
