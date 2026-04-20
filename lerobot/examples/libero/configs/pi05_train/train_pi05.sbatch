#!/usr/bin/env bash
#SBATCH --job-name=pi05_train
#SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/scratch/mdorazi
PROJDIR=/Skill_Boundary_Detector
cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

DATASET=libero_90 # libero_spatial_object
CONFIG_PATH=../outputs/pi05_${DATASET}_2/checkpoints/last/pretrained_model/train_config.json

RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
else
    echo "No checkpoint found. Starting fresh training."
fi

PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=lerobot/${DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/libero_dataset/${DATASET} \
    --policy.type=pi05 \
    --output_dir=../outputs/pi05_${DATASET}_2 \
    --job_name=pi05_PT_${DATASET} \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --save_freq=1000 \
    --steps=500000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=16 \
    ${RESUME_ARGS}

echo "DONE"
