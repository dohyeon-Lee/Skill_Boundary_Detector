#!/usr/bin/env bash
# Inputs:
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
# Reference model:
#   none
# Outputs:
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#
# Extract skill-level DINO tokens from prepared frame DINO features.
#SBATCH --job-name=skill_tokens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Extract skill DINO tokens"
echo "  frame DINO : ${DINO_FEATURE_DIR}"
echo "  skills     : ${SKILLSET_DIR}/skills"
echo "  output     : ${DINO_TOKENS_PATH}"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/extract_skill_dino_tokens.py \
    --frame_dino_dir "${DINO_FEATURE_DIR}" \
    --skills_dir     "${SKILLSET_DIR}/skills" \
    --output_path    "${DINO_TOKENS_PATH}" \
    --image_key      observation.images.image

echo "DONE -> ${DINO_TOKENS_PATH}"
