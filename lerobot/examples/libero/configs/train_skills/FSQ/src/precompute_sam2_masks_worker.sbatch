#!/usr/bin/env bash
# Inputs:
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
# Reference models:
#   SAM2         : sam2_checkpoint or {project_root}/models/sam2/sam2.1_hiera_large.pt
# Outputs:
#   SAM2 masks   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/sam2_masks
#
# SAM2 worker for FSQ patch flags.
#SBATCH --job-name=SAM2_masks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

: "${N_WORKERS:?}"

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

echo "Node: ${SLURMD_NODENAME:-local} | Worker: ${SLURM_ARRAY_TASK_ID:-0}/${N_WORKERS}"
nvidia-smi | head -20

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/precompute_sam2_masks.py \
    --skillset_dir           "${SKILLSET_DIR}/skills" \
    --dataset_dir            "${RAW_DATASET_DIR}" \
    --output_dir             "${SAM2_MASKS_DIR}" \
    --sam2_checkpoint        "${SAM2_CHECKPOINT}" \
    --sam2_config            configs/sam2.1/sam2.1_hiera_l.yaml \
    --image_key              observation.images.image \
    --device                 cuda \
    --points_per_side        50 \
    --max_mask_area_ratio    0.35 \
    --min_mask_area_ratio    0.001 \
    --min_stability_score    0.85 \
    --gripper_point          128 45 \
    --min_centroid_dist      30.0 \
    --min_visible_ratio      0.05 \
    --changed_top_crop_ratio 0.35 \
    --green_top_crop_ratio   0.35 \
    --changed_prop_back      2 \
    --image_size             224 \
    --patch_size             16 \
    --patch_hit_threshold    0.1 \
    --pool_size              "${DINO_PATCH_GRID}" \
    --num_workers            "${N_WORKERS}" \
    --worker_id              "${SLURM_ARRAY_TASK_ID:-0}"

echo "DONE worker ${SLURM_ARRAY_TASK_ID:-0}"
