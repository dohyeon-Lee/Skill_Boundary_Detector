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

# Slurm executes a spool copy of this file. Resolve repository paths from the
# build_data directory where the submitter invoked sbatch, not from BASH_SOURCE.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${SUBMIT_DIR}/src"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}"

if [ ! -f "${SCRIPT_DIR}/mark_skillset_complete.sbatch" ]; then
  echo "Cannot resolve build_data/src from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${SUBMIT_DIR}"
mkdir -p logs

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
if [ ! -f "${SKILLSET_DIR}/skillset_manifest.json" ]; then
  echo "Skillset manifest missing: ${SKILLSET_DIR}/skillset_manifest.json" >&2
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
  echo "skillset_mode=${SKILLSET_MODE}"
  echo "manifest=${SKILLSET_DIR}/skillset_manifest.json"
  echo "boundary_threshold_mode=${SKILLSET_BOUNDARY_THRESHOLD_MODE}"
  echo "boundary_threshold_scale=${SKILLSET_BOUNDARY_THRESHOLD_SCALE}"
  [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ] && \
    echo "global_threshold_path=${EFFECTIVE_GLOBAL_THRESHOLD_PATH}"
  [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ] && \
    echo "global_threshold_source=${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
  echo "completed_at=$(date -Iseconds)"
} > "${SKILLSET_DONE_PATH}"

echo "Skillset complete: ${SKILLSET_DONE_PATH} (${N_SKILLS} skills)"
