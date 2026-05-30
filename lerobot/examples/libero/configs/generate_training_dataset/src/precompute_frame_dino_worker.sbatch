#!/usr/bin/env bash
# Worker for run_frame_dino_parallel.sh.
# All paths and DINO hyperparameters are injected by the submitter.
#SBATCH --job-name=training_frame_DINO
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

: "${PROJECT_ROOT:?}" "${DATASET:?}" "${DATASET_DIR:?}" "${OUTPUT_DIR:?}" "${N_WORKERS:?}"

cd "${PROJECT_ROOT}/lerobot"
source "${PROJECT_ROOT}/.venv/bin/activate"

export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${PROJECT_ROOT}/../.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${PROJECT_ROOT}/../.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

IMAGE_KEYS=$(echo "${IMAGE_KEYS:-observation.images.image}" | tr ':' ',')
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "START FRAME DINO"
echo "Node       : ${SLURMD_NODENAME:-unknown}"
echo "Dataset    : ${DATASET_DIR}"
echo "Output     : ${OUTPUT_DIR}"
echo "Backbone   : ${VISUAL_BACKBONE}"
echo "Model      : ${IMAGE_MODEL_PATH}"
echo "Image keys : ${IMAGE_KEYS}"
echo "Patch grid : ${PATCH_GRID}"
echo "Worker     : ${WORKER_ID}/${N_WORKERS}"
nvidia-smi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/precompute_frame_dino_features.py \
    --dataset_dir      "${DATASET_DIR}" \
    --output_dir       "${OUTPUT_DIR}" \
    --image_keys       "${IMAGE_KEYS}" \
    --image_model_name "${IMAGE_MODEL_PATH}" \
    --image_size       "${IMAGE_SIZE}" \
    --patch_grid       "${PATCH_GRID}" \
    --n_patch_raw      "${N_PATCH_RAW}" \
    --batch_size       "${BATCH_SIZE}" \
    --dtype            "${DTYPE}" \
    --device           cuda \
    --num_workers      "${N_WORKERS}" \
    --worker_id        "${WORKER_ID}" \
    --wandb_project    "${WANDB_PROJECT:-}"

echo "DONE"
