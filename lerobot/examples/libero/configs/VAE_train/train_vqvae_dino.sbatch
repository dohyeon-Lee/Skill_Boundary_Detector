#!/usr/bin/env bash
#SBATCH --job-name=VQVAE_DINO_train

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
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

# hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir models/dinov3-vits16
# hf download facebook/dinov3-convnext-small-pretrain-lvd1689m --local-dir models/dinov3-convnext-small

set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate
# Avoid loading system cuDNN from /usr/local/cuda-* before PyTorch's bundled cuDNN.
export LD_LIBRARY_PATH=""

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

DATA=libero_90
LATENT_DIM=32 # 32 64 128
NUM_EMBEDDINGS=10 # 64 128
ENCODER_TYPE=gru
DECODER_RNN_TYPE=mlp
ZERO_START_EEF=false # true | false
END_THRESHOLD=0.5
END_POS_WEIGHT=1
COMMITMENT_COST=0.1   # default 0.25;
VISUAL_BACKBONE=dinov3_vits16 # dinov2_small
SKILLSET=${DATA}_skillset

# -- exp1 (default) -- #
# DELTA_LOSS_WEIGHT=1.0  # default 1.0; reconstruction loss weight
# END_LOSS_WEIGHT=10.0   # default 10.0; end-signal BCE loss weight
# -- exp2 -- #
# DELTA_LOSS_WEIGHT=10.0  # default 1.0; reconstruction loss weight
# END_LOSS_WEIGHT=100.0   # default 10.0; end-signal BCE loss weight
# -- exp3 -- #
DELTA_LOSS_WEIGHT=10.0  # default 1.0; reconstruction loss weight
END_LOSS_WEIGHT=10.0   # default 10.0; end-signal BCE loss weight
# -- exp4 -- #
# DELTA_LOSS_WEIGHT=10.0  # default 1.0; reconstruction loss weight
# END_LOSS_WEIGHT=5.0   # default 10.0; end-signal BCE loss weight

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then
  EEF_MODE=zero
  ZERO_START_ARG="--zero-start-eef"
else
  EEF_MODE=abs
  ZERO_START_ARG="--no-zero-start-eef"
fi

EXP_NAME="${DATA}_vqvae_${IMAGE_EXP_TAG}_${ENCODER_TYPE}_${DECODER_RNN_TYPE}_${EEF_MODE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_${COMMITMENT_COST}_exp3"
OUTPUT_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

if [ ! -d "${IMAGE_MODEL_PATH}" ]; then
  echo "Missing visual backbone weights at ${IMAGE_MODEL_PATH}"
  echo "Download with:"
  echo "  huggingface-cli download ${IMAGE_MODEL_REPO} --local-dir ${IMAGE_MODEL_PATH}"
  echo "DINOv3 models may require Hugging Face login and license approval."
  exit 1
fi
if [ ! -f "${IMAGE_FEATURES_PATH}" ]; then
  echo "Missing precomputed visual features at ${IMAGE_FEATURES_PATH}"
  echo "Create them with:"
  echo "  VISUAL_BACKBONE=${VISUAL_BACKBONE} bash examples/libero/configs/VAE_train/run_dino_parallel.sh"
  exit 1
fi

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
  LATEST_CKPT=$(ls -v "${OUTPUT_DIR}"/spline_vqae_epoch*.pt 2>/dev/null | tail -1 || true)
  if [ -n "${LATEST_CKPT}" ]; then
    RESUME_ARG="--resume_from ${LATEST_CKPT}"
    echo "Resuming from ${LATEST_CKPT}"
  else
    echo "No checkpoint found, training from scratch"
  fi
else
  echo "No checkpoint found, training from scratch"
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/train_vae.py \
    --skills_dir ${HOMEDIR}${PROJDIR}/libero_dataset/${SKILLSET} \
    --dataset_dir ${HOMEDIR}${PROJDIR}/libero_dataset/${DATA} \
    --output_dir ${OUTPUT_DIR} \
    --latent_dim ${LATENT_DIM} \
    --num_embeddings ${NUM_EMBEDDINGS} \
    --encoder_type ${ENCODER_TYPE} \
    --decoder_rnn_type ${DECODER_RNN_TYPE} \
    --commitment_cost ${COMMITMENT_COST} \
    --hidden_dim 256 \
    --num_layers 2 \
    --n_control 30 \
    --epochs 3000 \
    --lr 3e-4 \
    --batch_size 4 \
    --end_threshold ${END_THRESHOLD} \
    --end_pos_weight ${END_POS_WEIGHT} \
    --delta_loss_weight ${DELTA_LOSS_WEIGHT} \
    --end_loss_weight ${END_LOSS_WEIGHT} \
    --checkpoint_every 500 \
    ${ZERO_START_ARG} \
    --image_features_path ${IMAGE_FEATURES_PATH} \
    --image_model_name ${IMAGE_MODEL_PATH} \
    --image_size 224 \
    --wandb_project VAE_train \
    --wandb_run_name ${EXP_NAME} \
    ${RESUME_ARG}

echo "DONE ${EXP_NAME}"
