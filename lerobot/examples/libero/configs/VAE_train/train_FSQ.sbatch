#!/usr/bin/env bash
#SBATCH --job-name=FSQ_train
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
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
DERIVED_DATA_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data
FSQ_PRECOMPUTE_DIR="${DERIVED_DATA_DIR}/${DATA}_for_FSQ"
SKILLSET_DIR="${FSQ_PRECOMPUTE_DIR}/${SKILLSET}"

DECODER_MODE=single_step  # single_step | chunk
DECODER_IMAGE_MODE=dino_flags  # dino_only | dino_flags
CHUNK_SIZE=10
FSQ_LEVELS=(3 3 3)
ZERO_START_EEF=false
END_THRESHOLD=0.5
END_POS_WEIGHT=1
END_RANDOM_P=5
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

FSQ_TAG="fsq$(IFS=; echo "${FSQ_LEVELS[*]}")"
FSQ_CODEBOOK_SIZE=1
for level in "${FSQ_LEVELS[@]}"; do
  FSQ_CODEBOOK_SIZE=$((FSQ_CODEBOOK_SIZE * level))
done
if [ "${DECODER_MODE}" = "chunk" ]; then
  MODE_TAG="chunk"
else
  MODE_TAG="single_step"
fi
EXP_NAME="${DATA}_${FSQ_TAG}_${IMAGE_EXP_TAG}_${EEF_MODE}_${MODE_TAG}_${DECODER_IMAGE_MODE}"
OUTPUT_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_FSQ/${EXP_NAME}"

DINO_TOKENS_PATH="${FSQ_PRECOMPUTE_DIR}/${IMAGE_FEATURE_TAG}_tokens.npz"

# SAM2 patch flags — run_dino_parallel.sh writes per-skill temporal masks, then
# merges them into temporal patch_flags.npz for faster FSQ training.
SAM2_MASKS_DIR="${FSQ_PRECOMPUTE_DIR}/sam2_masks"
SAM2_FLAGS_PATH="${FSQ_PRECOMPUTE_DIR}/patch_flags.npz"
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "ERROR: DINO token features not found at ${DINO_TOKENS_PATH}"
  echo "Run VAE_train/run_dino_parallel.sh first."
  exit 1
fi

if [ -f "${SAM2_FLAGS_PATH}" ]; then
  SAM2_INPUT="${SAM2_FLAGS_PATH}"
else
  SAM2_INPUT="${SAM2_MASKS_DIR}"
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
    --skills_dir              ${SKILLSET_DIR}/skills \
    --dino_features           ${DINO_TOKENS_PATH} \
    --sam2_masks_dir          ${SAM2_INPUT} \
    --output_dir              ${OUTPUT_DIR} \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --decoder_output_mode     ${DECODER_MODE} \
    --decoder_image_mode      ${DECODER_IMAGE_MODE} \
    --chunk_size              ${CHUNK_SIZE} \
    --hidden_dim              256 \
    --num_layers              2 \
    --n_control               30 \
    --feat_dim                384 \
    --n_tokens                65 \
    --image_model_name        ${IMAGE_MODEL_PATH} \
    --epochs                  5000 \
    --lr                      3e-4 \
    --batch_size              64 \
    --end_threshold           ${END_THRESHOLD} \
    --end_pos_weight          ${END_POS_WEIGHT} \
    --end_random_p            ${END_RANDOM_P} \
    --delta_loss_weight       ${DELTA_LOSS_WEIGHT} \
    --end_loss_weight         ${END_LOSS_WEIGHT} \
    --max_length              200 \
    --checkpoint_every        500 \
    ${ZERO_START_ARG} \
    --wandb_project           VAE_train \
    --wandb_run_name          ${EXP_NAME} \
    ${RESUME_ARG}

echo "DONE ${EXP_NAME}"
