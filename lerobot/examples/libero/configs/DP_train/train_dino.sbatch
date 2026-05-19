#!/usr/bin/env bash
#SBATCH --job-name=DP_DINO_train

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

DATASET=libero_90
DATASET_ROOT=libero_dataset

N_OBS_STEPS=10
N_ACTION_STEPS=16
HORIZON=16

VISUAL_BACKBONE=dinov3_vits16
PATCH_GRID=8
DINO_FEATURE_DIR=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET}_data/${DATASET}_DINO/${VISUAL_BACKBONE}_pg${PATCH_GRID}
DINO_IMAGE_KEYS='["observation.images.image"]'

TRAIN_CONFIG_PATH=${HOMEDIR}${PROJDIR}/lerobot/examples/libero/configs/DP_train/train_dp_${DATASET}.yaml
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/dp_${DATASET}_dino_${VISUAL_BACKBONE}_pg${PATCH_GRID}_obs${N_OBS_STEPS}_horizon${HORIZON}
WANDB_PROJECT=DP_train
JOB_NAME=DP_${DATASET}_dino_${VISUAL_BACKBONE}_pg${PATCH_GRID}_obs${N_OBS_STEPS}_horizon${HORIZON}
SAVE_FREQ=5000

BATCH_SIZE=64
NUM_WORKERS=4

if [ ! -d "${DINO_FEATURE_DIR}" ]; then
  echo "DINO feature dir not found: ${DINO_FEATURE_DIR}" >&2
  echo "Run configs/DP_train/run_frame_dino_parallel.sh first." >&2
  exit 1
fi

RESUME_FLAG=""
CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  lerobot-train \
  --config_path=${TRAIN_CONFIG_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --wandb.enable=true \
  --wandb.project=${WANDB_PROJECT} \
  --save_freq=${SAVE_FREQ} \
  --policy.n_obs_steps=${N_OBS_STEPS} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.horizon=${HORIZON} \
  --policy.use_dino_features=true \
  --policy.dino_feature_dir=${DINO_FEATURE_DIR} \
  --policy.dino_image_keys=${DINO_IMAGE_KEYS} \
  --policy.dino_patch_grid=${PATCH_GRID} \
  --policy.dino_feature_dim=384 \
  --policy.dino_visual_feature_dim=512 \
  --policy.dino_conv_hidden_dim=256 \
  --dataset.repo_id=lerobot/${DATASET} \
  --dataset.root=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET} \
  --dataset.video_backend=pyav \
  --batch_size=${BATCH_SIZE} \
  --num_workers=${NUM_WORKERS} \
  ${RESUME_FLAG}

echo "DONE"
