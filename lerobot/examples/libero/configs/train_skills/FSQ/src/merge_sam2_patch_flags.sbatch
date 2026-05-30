#!/usr/bin/env bash
# Inputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
#   SAM2 masks   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/sam2_masks
# Reference model:
#   none
# Outputs:
#   SAM2 flags   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/patch_flags.npz
#
# Merge per-skill SAM2 masks into one patch_flags npz.
#SBATCH --job-name=merge_SAM2_flags
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

mkdir -p "$(dirname "${SAM2_FLAGS_PATH}")"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero" \
  python -u examples/libero/merge_sam2_patch_flags.py \
    --skills_dir     "${SKILLSET_DIR}/skills" \
    --sam2_masks_dir "${SAM2_MASKS_DIR}" \
    --output_path    "${SAM2_FLAGS_PATH}" \
    --n_patches      $((DINO_PATCH_GRID * DINO_PATCH_GRID))

echo "DONE -> ${SAM2_FLAGS_PATH}"
