#!/usr/bin/env bash
# Inputs:
#   dataset    : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   skillset   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
#   raw video  : {project_root}/{dataset_root}/{target_dataset}/videos
# Reference models:
#   SAM3       : {project_root}/models/sam3
#   SAM2       : sam2_checkpoint or {project_root}/models/sam2/sam2.1_hiera_large.pt
# Outputs:
#   PNGs       : FSQ_data_eval/outputs/sam3_seeded_motion by default
#
# Run SAM3-seeded motion visualization on one allocated GPU.
#SBATCH --job-name=SAM3_motion_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
COMMON_SRC_DIR="${SUBMIT_DIR}/../src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/../train_skills_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

echo "START SAM3 SEEDED MOTION EVAL"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "Dataset   : ${TARGET_DATASET}"
echo "Skillset  : ${SKILLSET_DIR}/skills"
echo "Raw data  : ${RAW_DATASET_DIR}"
echo "Args      : ${EVAL_ARGS_QUOTED:-}"
nvidia-smi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi
if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Raw dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/lerobot/src"
eval "python -u examples/libero/configs/train_skills/FSQ_data_eval/eval_SAM3_seeded_motion.py ${EVAL_ARGS_QUOTED:-}"

echo "DONE"
