#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_FT

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

# ── Fine-tune dataset / base checkpoint ─────────────────────────────────────
# FT_DATASET must have the same SkillVLA columns as libero_90_skillvla.
FT_DATASET=libero_10_op1_10_skillvla
FT_DATASET_ROOT=libero_dataset
FT_DATASET_SOURCE=${FT_DATASET%_skillvla}
FT_DATASET_DIR=${HOMEDIR}${PROJDIR}/${FT_DATASET_ROOT}/${FT_DATASET_SOURCE}_data/${FT_DATASET}

# SkillVLA checkpoint trained on libero_90.
PT_RUN_NAME=PT_libero_90_skillvla_true_embs_lat128_vq128
PT_CHECKPOINT=025000
BASE_POLICY=${HOMEDIR}${PROJDIR}/outputs/${PT_RUN_NAME}/checkpoints/${PT_CHECKPOINT}/pretrained_model

# VQ-VAE decoder used by the base SkillVLA run. Keep this matched to PT.
LATENT_DIM=128
NUM_EMBEDDINGS=128
VQVAE_STEP=1500
SKILLSET=libero_90_skillset
VQVAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/libero_90_vqvae_dinov3_vits16_gru_mlp_zero_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_0.1_exp3/spline_vqae_epoch$(printf '%04d' ${VQVAE_STEP}).pt

if [ ! -f "${BASE_POLICY}/model.safetensors" ]; then
    echo "Missing base SkillVLA checkpoint: ${BASE_POLICY}"
    exit 1
fi

if [ ! -f "${VQVAE_CKPT}" ]; then
    echo "Missing VQ-VAE checkpoint: ${VQVAE_CKPT}"
    exit 1
fi

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

# Fine-tuning follows pi05_FT: keep the vision encoder frozen by default.
FREEZE_VISION_ENCODER=true
# true: freeze the full Paligemma VLM, while leaving the action expert,
# skill predictor, and SkillVLA projection/decoder modules controlled by
# their own freeze flags.
TRAIN_EXPERT_ONLY=false

# true: action expert attends to image/state prefix tokens, not language tokens.
BLOCK_LANG=true

# context: Paligemma prefix-only forward 이후 contextual hidden
# embs: embed_prefix 직후 raw prefix token embedding
SP_PREFIX_SOURCE=embs

# Scheduled sampling is restarted for this fine-tuning run:
# full teacher forcing at step 0, then gradually uses predicted tokens.
PREDICTED_LATENT_SAMPLING=true
PREDICTED_LATENT_START_STEP=0
PREDICTED_LATENT_RAMP_STEPS=30000
PREDICTED_LATENT_MAX_PROB=1.0

RUN_NAME=${PT_CHECKPOINT}PT_${FT_DATASET}FT_${PREDICTED_LATENT_SAMPLING}_${SP_PREFIX_SOURCE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_${TRAIN_EXPERT_ONLY}
OUTPUT_DIR=../outputs/FT_${RUN_NAME}
CONFIG_PATH=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json

RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint found. Removing existing output dir and starting fresh."
    rm -rf "${OUTPUT_DIR}"
else
    echo "No checkpoint found. Starting fresh fine-tuning."
fi

# ── SkillVLA fine-tuning from libero_90 checkpoint ──────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${FT_DATASET} \
    --dataset.root=${FT_DATASET_DIR} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --policy.pretrained_path=${BASE_POLICY} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_Finetune \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=${FREEZE_VISION_ENCODER} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
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
    --save_freq=2000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}

echo "DONE"
