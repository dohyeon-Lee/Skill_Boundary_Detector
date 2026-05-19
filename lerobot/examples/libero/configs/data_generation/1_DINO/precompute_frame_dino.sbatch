#!/usr/bin/env bash
#SBATCH --job-name=DP_frame_DINO

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
# --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

HOMEDIR=${HOMEDIR:-/data2/dohyeon}
PROJDIR=${PROJDIR:-/SBD}

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi

# Raw LeRobot dataset to precompute.
DATASET=${DATASET:-libero_90}
DATASET_ROOT=${DATASET_ROOT:-libero_dataset}

# Comma-separated camera keys, or "all".
# IMAGE_KEYS is exported with colons instead of commas to avoid SLURM --export parsing issues
IMAGE_KEYS=$(echo "${IMAGE_KEYS:-observation.images.image}" | tr ':' ',')

# Visual backbone preset.
# Supported here: dinov3_vits16, dinov2_small
VISUAL_BACKBONE=${VISUAL_BACKBONE:-dinov3_vits16}
case "${VISUAL_BACKBONE}" in
  dinov3_vits16|dinov3_vits14)
    IMAGE_MODEL_DIR=dinov3-vits16
    IMAGE_FEATURE_TAG=dinov3_vits16
    N_PATCH_RAW=${N_PATCH_RAW:-196}
    ;;
  dinov2_small)
    IMAGE_MODEL_DIR=dinov2-small
    IMAGE_FEATURE_TAG=dinov2_small
    N_PATCH_RAW=${N_PATCH_RAW:-256}
    ;;
  *)
    echo "Unknown VISUAL_BACKBONE=${VISUAL_BACKBONE}" >&2
    exit 1
    ;;
esac

IMAGE_MODEL_PATH=${HOMEDIR}${PROJDIR}/models/${IMAGE_MODEL_DIR}

PATCH_GRID=${PATCH_GRID:-8}
IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-1024}
DTYPE=${DTYPE:-float16}
NUM_WORKERS=${N_WORKERS:-1}
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}
WANDB_PROJECT=${WANDB_PROJECT:-}

DATASET_DIR=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET}
DERIVED_DATA_DIR=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET}_data
OUTPUT_DIR=${DERIVED_DATA_DIR}/${DATASET}_DINO/${IMAGE_FEATURE_TAG}_pg${PATCH_GRID}

if [ ! -d "${DATASET_DIR}" ]; then
  echo "Dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

if [ ! -d "${IMAGE_MODEL_PATH}" ]; then
  echo "Image model not found: ${IMAGE_MODEL_PATH}" >&2
  exit 1
fi

echo "Dataset      : ${DATASET_DIR}"
echo "Output       : ${OUTPUT_DIR}"
echo "Image keys   : ${IMAGE_KEYS}"
echo "Backbone     : ${VISUAL_BACKBONE} (${IMAGE_MODEL_PATH})"
echo "Patch grid   : ${PATCH_GRID}x${PATCH_GRID}, raw patches=${N_PATCH_RAW}"
echo "Worker       : ${WORKER_ID}/${NUM_WORKERS}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/precompute_frame_dino_features.py \
    --dataset_dir      ${DATASET_DIR} \
    --output_dir       ${OUTPUT_DIR} \
    --image_keys       ${IMAGE_KEYS} \
    --image_model_name ${IMAGE_MODEL_PATH} \
    --image_size       ${IMAGE_SIZE} \
    --patch_grid       ${PATCH_GRID} \
    --n_patch_raw      ${N_PATCH_RAW} \
    --batch_size       ${BATCH_SIZE} \
    --dtype            ${DTYPE} \
    --device           cuda \
    --num_workers      ${NUM_WORKERS} \
    --worker_id        ${WORKER_ID} \
    --wandb_project    "${WANDB_PROJECT}"

echo "DONE"
