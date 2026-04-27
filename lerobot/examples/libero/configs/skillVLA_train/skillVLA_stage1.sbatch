#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_s1

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
DATASET=libero_90_skillvla          # add_skill_latents_to_dataset.py 로 만든 데이터셋
DATASET_ROOT=libero_dataset
PI05_CKPT_PATH=${HOMEDIR}${PROJDIR}/outputs/pi05_libero_90/checkpoints/070000/pretrained_model

LR=2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4
NUM_GPUS=1

OUTPUT_DIR=../outputs/skillVLA_s1_${DATASET}
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

# ── Stage 1: VLM + action expert 학습, skill predictor frozen ─────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=skillVLA_s1_${DATASET} \
    --policy.pretrained_path=${PI05_CKPT_PATH} \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=SkillVLA_train \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.training_stage=1 \
    --policy.block_lang_to_action=true \
    --policy.skill_latent_dim=16 \
    --policy.skill_predictor_loss_weight=1.0 \
    --save_freq=10000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}

echo "DONE"
