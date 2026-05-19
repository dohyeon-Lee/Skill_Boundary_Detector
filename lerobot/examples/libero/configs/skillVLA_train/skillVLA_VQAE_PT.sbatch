#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_vqvae_train

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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

# ── Dataset / checkpoints ────────────────────────────────────────────────────
# Dataset must have skill_sequence/skill_index columns from add_skill_latents_to_dataset.py.
DATASET=libero_90_skillvla
DATASET_ROOT=libero_dataset
DATASET_SOURCE=${DATASET%_skillvla}
DATASET_DIR=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET_SOURCE}_data/${DATASET}
PI_BASE=lerobot/pi05_base

LATENT_DIM=128
NUM_EMBEDDINGS=128
VQVAE_STEP=1500
SKILLSET=libero_90_skillset

VQVAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/libero_90_vqvae_dinov3_vits16_gru_mlp_zero_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_0.1_exp3/spline_vqae_epoch$(printf '%04d' ${VQVAE_STEP}).pt

# ── Optimization ─────────────────────────────────────────────────────────────
LR=2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4
NUM_GPUS=1

# ── SkillVLA losses / boundary randomization ────────────────────────────────
SP_LOSS_WEIGHT=1.0
SKILL_DECODER_LOSS_WEIGHT=1.0
SKILL_DECODER_END_POS_WEIGHT=1.0
SKILL_DECODER_END_THRESHOLD=0.5
SKILL_DECODER_STATE_INDICES="[0,1,2,3,4,5]"
FREEZE_VAE_DECODER=false
INFERENCE_SKILL_MAX_ORDER=20
INFERENCE_SKILL_MAX_LENGTH=200
SKILL_BOUNDARY_RANDOM_P=10

# true: action expert attends to image/state prefix tokens, not language tokens.
BLOCK_LANG=true

# context: Paligemma prefix-only forward 이후 contextual hidden
# embs: embed_prefix 직후 raw prefix token embedding
SP_PREFIX_SOURCE=embs

# Scheduled sampling: linearly ramp from GT token → predicted token
PREDICTED_LATENT_SAMPLING=true
PREDICTED_LATENT_START_STEP=0
PREDICTED_LATENT_RAMP_STEPS=50000
PREDICTED_LATENT_MAX_PROB=1.0

RUN_NAME=${DATASET}_${PREDICTED_LATENT_SAMPLING}_${SP_PREFIX_SOURCE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}
OUTPUT_DIR=../outputs/PT_${RUN_NAME}
CONFIG_PATH=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json

RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint found. Removing existing output dir and starting fresh."
    rm -rf "${OUTPUT_DIR}"
else
    echo "No checkpoint found. Starting fresh training."
fi

# ── SkillVLA VQ-VAE-end training ─────────────────────────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${DATASET} \
    --dataset.root=${DATASET_DIR} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --policy.pretrained_path=${PI_BASE} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.block_lang_to_action=${BLOCK_LANG} \
    --policy.vae_decoder_path=${VQVAE_CKPT} \
    --policy.freeze_vae_decoder=${FREEZE_VAE_DECODER} \
    --policy.skill_latent_dim=${LATENT_DIM} \
    --policy.skill_predictor_num_embeddings=${NUM_EMBEDDINGS} \
    --policy.skill_predictor_prefix_source=${SP_PREFIX_SOURCE} \
    --policy.skill_predictor_loss_weight=${SP_LOSS_WEIGHT} \
    --policy.skill_decoder_loss_weight=${SKILL_DECODER_LOSS_WEIGHT} \
    --policy.skill_decoder_end_pos_weight=${SKILL_DECODER_END_POS_WEIGHT} \
    --policy.skill_decoder_end_threshold=${SKILL_DECODER_END_THRESHOLD} \
    --policy.skill_decoder_state_indices=${SKILL_DECODER_STATE_INDICES} \
    --policy.inference_skill_max_order=${INFERENCE_SKILL_MAX_ORDER} \
    --policy.inference_skill_max_length=${INFERENCE_SKILL_MAX_LENGTH} \
    --policy.skill_boundary_random_p=${SKILL_BOUNDARY_RANDOM_P} \
    --policy.predicted_latent_sampling=${PREDICTED_LATENT_SAMPLING} \
    --policy.predicted_latent_start_step=${PREDICTED_LATENT_START_STEP} \
    --policy.predicted_latent_ramp_steps=${PREDICTED_LATENT_RAMP_STEPS} \
    --policy.predicted_latent_max_prob=${PREDICTED_LATENT_MAX_PROB} \
    --save_freq=5000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}


echo "DONE"
