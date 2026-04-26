#!/usr/bin/env bash
#SBATCH --job-name=pi05_FT

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
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

# current RLLAB code
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

FT_DATASET=libero_10_op1_50
FT_DATASET_LOC=libero_small_dataset

# FT_DATASET=libero_10
# FT_DATASET_LOC=libero_dataset

PT_DATASET=libero_90
PT_check=070000
LR=2.5e-05        # default: 2.5e-05
NUM_WORKERS=4     # shm 에러 시 0으로 낮출 것

CONFIG_PATH=../outputs/pi05_${PT_check}PT_${FT_DATASET}FT/checkpoints/last/pretrained_model/train_config.json
PT_MODEL_PATH=${HOMEDIR}${PROJDIR}/outputs/pi05_${PT_DATASET}/checkpoints/${PT_check}/pretrained_model
RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
else
    echo "No checkpoint found. Starting fresh training."
fi

PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=lerobot/${FT_DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/${FT_DATASET_LOC}/${FT_DATASET} \
    --policy.type=pi05 \
    --output_dir=../outputs/pi05_${PT_check}PT_${FT_DATASET}FT \
    --job_name=pi05_FT_${PT_DATASET}${PT_check}_${FT_DATASET} \
    --policy.pretrained_path=${PT_MODEL_PATH} \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_Finetune \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --save_freq=5000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=32 \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}

echo "DONE"
