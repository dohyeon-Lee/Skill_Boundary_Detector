#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_PT

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

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATASET=libero_90_skillvla
DATASET_ROOT=libero_dataset
PI_BASE=lerobot/pi05_base
VAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat64_spline/spline_vae_epoch10000.pt

LR=2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4
NUM_GPUS=1

# boundary 프레임 비율(~1/chunk_len) 보정. 10~40 사이로 튜닝
SP_LOSS_WEIGHT=1.0

# true: action expert가 image token만 attend (language 차단)
# false: pi05 기본 동작 (language + obs 모두 attend)
BLOCK_LANG=true

# Scheduled sampling for the latent fed to the action expert and VAE decoder.
# false: teacher-forced GT z only. true: linearly replace GT z with skill-predictor z.
PREDICTED_LATENT_SAMPLING=false
PREDICTED_LATENT_START_STEP=0
PREDICTED_LATENT_RAMP_STEPS=50000
PREDICTED_LATENT_MAX_PROB=1.0
PREDICTED_LATENT_DETACH=true

OUTPUT_DIR=../outputs/PT_${DATASET}_coupled_${PREDICTED_LATENT_SAMPLING}
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

# ── Stage 3: joint 학습 — teacher forcing + path3 (sp_loss → SP → VLM) ────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${DATASET}_PT_coupled_${PREDICTED_LATENT_SAMPLING} \
    --policy.pretrained_path=${PI_BASE} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.block_lang_to_action=${BLOCK_LANG} \
    --policy.vae_decoder_path=${VAE_CKPT} \
    --policy.skill_latent_dim=64 \
    --policy.skill_predictor_state_dim=8 \
    --policy.skill_predictor_loss_weight=${SP_LOSS_WEIGHT} \
    --policy.predicted_latent_sampling=${PREDICTED_LATENT_SAMPLING} \
    --policy.predicted_latent_start_step=${PREDICTED_LATENT_START_STEP} \
    --policy.predicted_latent_ramp_steps=${PREDICTED_LATENT_RAMP_STEPS} \
    --policy.predicted_latent_max_prob=${PREDICTED_LATENT_MAX_PROB} \
    --policy.predicted_latent_detach=${PREDICTED_LATENT_DETACH} \
    --save_freq=5000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}


echo "DONE"
