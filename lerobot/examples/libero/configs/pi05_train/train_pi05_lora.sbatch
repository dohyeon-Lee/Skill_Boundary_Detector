#!/usr/bin/env bash
#SBATCH --job-name=pi05_lora_train
#SBATCH --partition=suma_a6000,asus_6000ada #asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=lerobot/libero_90 \
    --dataset.root=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
    --policy.type=pi05 \
    --output_dir=../outputs/pi05_lora_libero_90 \
    --job_name=pi05_lora_PT_libero90 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_lora=true \
    --policy.lora_rank=16 \
    --policy.lora_alpha=16 \
    --policy.lora_dropout=0.05 \
    --policy.push_to_hub=false \
    --save_freq=500 \
    --steps=30000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=32 \
    --log_freq=50 \
    --resume=true \
    --config_path=../outputs/pi05_lora_libero_90/checkpoints/last/pretrained_model/train_config.json


echo "DONE"
