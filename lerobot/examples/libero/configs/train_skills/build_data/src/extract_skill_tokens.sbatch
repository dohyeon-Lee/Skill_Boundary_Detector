#!/usr/bin/env bash
# Inputs:
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
# Reference model:
#   none
# Outputs:
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#
# Extract skill-level DINO tokens from prepared frame DINO features.
#SBATCH --job-name=skill_tokens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# mem: each camera builds one ~27GB (pg8) float16 array in RAM (list + concatenate copy ≈ 2x). 32G
# caused swap thrashing (28→3 it/s) and the 30-min wall killed it at 94%. 96G holds it without swap.
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Extract skill DINO tokens"
echo "  frame DINO (FSQ) : ${FSQ_DINO_FEATURE_DIR}"
echo "  skills           : ${SKILLSET_DIR}/skills"
echo "  use third / wrist: ${FSQ_TERMINATOR_USE_THIRD} / ${FSQ_TERMINATOR_USE_WRIST}"

# Extract only the cameras the FSQ uses. wrist-only (third=false, wrist=true) → no 3rd tokens.
if [ "${FSQ_TERMINATOR_USE_THIRD}" = "true" ]; then
  echo "  output 3rd       : ${DINO_TOKENS_PATH}"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/extract_skill_dino_tokens.py \
      --frame_dino_dir "${FSQ_DINO_FEATURE_DIR}" \
      --skills_dir     "${SKILLSET_DIR}/skills" \
      --output_path    "${DINO_TOKENS_PATH}" \
      --image_key      observation.images.image
fi

if [ "${FSQ_TERMINATOR_USE_WRIST}" = "true" ]; then
  echo "  output wrist     : ${DINO_TOKENS_WRIST_PATH}"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/extract_skill_dino_tokens.py \
      --frame_dino_dir "${FSQ_DINO_FEATURE_DIR}" \
      --skills_dir     "${SKILLSET_DIR}/skills" \
      --output_path    "${DINO_TOKENS_WRIST_PATH}" \
      --image_key      observation.images.wrist_image
fi

echo "DONE -> ${DINO_TOKENS_PATH} (third=${FSQ_TERMINATOR_USE_THIRD}) / ${DINO_TOKENS_WRIST_PATH} (wrist=${FSQ_TERMINATOR_USE_WRIST})"
