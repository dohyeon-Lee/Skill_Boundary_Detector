#!/usr/bin/env bash
# Inputs:
#   SAM2 flags   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/patch_flags.npz
#   SAM2 masks   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/sam2_masks
# Reference model:
#   none
# Outputs:
#   removes      : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/sam2_masks
#
# Remove intermediate SAM2 mask files after patch_flags.npz was merged.
#SBATCH --job-name=cleanup_SAM2_masks
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
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

if [ ! -f "${SAM2_FLAGS_PATH}" ]; then
  echo "Refusing to remove SAM2 masks because patch flags are missing: ${SAM2_FLAGS_PATH}" >&2
  exit 1
fi

if [ -d "${SAM2_MASKS_DIR}" ]; then
  echo "Removing intermediate SAM2 masks: ${SAM2_MASKS_DIR}"
  rm -rf "${SAM2_MASKS_DIR}"
else
  echo "SAM2 masks dir already absent: ${SAM2_MASKS_DIR}"
fi

echo "DONE"
