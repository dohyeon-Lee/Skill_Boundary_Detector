#!/usr/bin/env bash
#SBATCH --job-name=FSQ_train
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
export LD_LIBRARY_PATH=""

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA=libero_90
DATADIR=libero_dataset
VISUAL_BACKBONE=dinov3_vits16
SKILLSET=${DATA}_skillset

DECODER_MODE=chunk        # single_step | chunk
CHUNK_SIZE=10
ZERO_START_EEF=true
END_THRESHOLD=0.5
END_POS_WEIGHT=1
DELTA_LOSS_WEIGHT=10.0
END_LOSS_WEIGHT=10.0

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then
  EEF_MODE=zero
  ZERO_START_ARG="--zero_start_eef"
else
  EEF_MODE=abs
  ZERO_START_ARG="--no-zero_start_eef"
fi

# FSQ 고정: levels=555, codebook=125
FSQ_TAG="fsq555"
MODE_TAG="${DECODER_MODE}K${CHUNK_SIZE}"
EXP_NAME="${DATA}_${FSQ_TAG}_${IMAGE_EXP_TAG}_${EEF_MODE}_${MODE_TAG}"
OUTPUT_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

# DINO full-token features (T × n_tokens × feat_dim) — precomputed by precompute_dino_features.py
# NOTE: precompute_dino_features.py must be run in token mode (outputs all patch tokens, not CLS only)
DINO_TOKENS_PATH="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_${VISUAL_BACKBONE}_tokens.npz"

# SAM2 masks output directory — produced by run_dino_parallel.sh + precompute_sam2_masks_worker.sbatch
SAM2_MASKS_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_FSQ"
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "ERROR: DINO token features not found at ${DINO_TOKENS_PATH}"
  echo "Run precompute_dino_features.py in token mode first."
  exit 1
fi

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
  LATEST_CKPT=$(ls -v "${OUTPUT_DIR}"/FSQ_epoch*.pt 2>/dev/null | tail -1 || true)
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
  python -u examples/libero/train_FSQ.py \
    --skills_dir              ${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET}/skills \
    --dino_features           ${DINO_TOKENS_PATH} \
    --sam2_masks_dir          ${SAM2_MASKS_DIR} \
    --output_dir              ${OUTPUT_DIR} \
    --fsq_levels              5 5 5 \
    --decoder_output_mode     ${DECODER_MODE} \
    --chunk_size              ${CHUNK_SIZE} \
    --hidden_dim              256 \
    --num_layers              2 \
    --n_control               30 \
    --feat_dim                384 \
    --n_tokens                65 \
    --epochs                  5000 \
    --lr                      3e-4 \
    --batch_size              64 \
    --end_threshold           ${END_THRESHOLD} \
    --end_pos_weight          ${END_POS_WEIGHT} \
    --delta_loss_weight       ${DELTA_LOSS_WEIGHT} \
    --end_loss_weight         ${END_LOSS_WEIGHT} \
    --max_length              200 \
    --checkpoint_every        500 \
    ${ZERO_START_ARG} \
    --wandb_project           VAE_train \
    --wandb_run_name          ${EXP_NAME} \
    ${RESUME_ARG}

echo "DONE ${EXP_NAME}"
