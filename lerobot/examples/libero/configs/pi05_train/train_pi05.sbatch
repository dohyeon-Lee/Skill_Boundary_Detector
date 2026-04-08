#!/usr/bin/env bash
#SBATCH --job-name=pi05_train
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
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

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=lerobot/libero_spatial_object \
    --dataset.root=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_spatial_object \
    --policy.type=pi05 \
    --output_dir=../outputs/pi05_libero_spatial_object \
    --job_name=pi05_training_SO \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --save_freq=1000 \
    --steps=30000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=8 \
    --resume=true \
    --config_path=../outputs/pi05_libero_spatial_object/checkpoints/013000/pretrained_model/train_config.json

# lerobot-train \
#     --dataset.repo_id=lerobot/libero_90 \
#     --dataset.root=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
#     --policy.type=pi05 \
#     --output_dir=../outputs/pi05_libero_90 \
#     --job_name=pi05_training_90 \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --policy.push_to_hub=false \
#     --save_freq=1000 \
#     --steps=30000 \
#     --policy.device=cuda \
#     --dataset.video_backend=pyav \
#     --batch_size=32
#     # --resume=true

# for RLLAB cluster
# PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
#     --dataset.repo_id=lerobot/libero_spatial_object \
#     --dataset.root=/data2/dohyeon/SBD/libero_dataset/libero_spatial_object \
#     --policy.type=pi05 \
#     --output_dir=../outputs/pi05_libero_spatial_object \
#     --job_name=pi05_training_SO \
#     --policy.pretrained_path=lerobot/pi05_base \
#     --policy.compile_model=true \
#     --policy.gradient_checkpointing=true \
#     --wandb.enable=true \
#     --policy.dtype=bfloat16 \
#     --policy.freeze_vision_encoder=false \
#     --policy.train_expert_only=false \
#     --policy.push_to_hub=false \
#     --save_freq=1000 \
#     --steps=30000 \
#     --policy.device=cuda \
#     --dataset.video_backend=pyav \
#     --batch_size=8

# # cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
# # wandb sync outputs/pi05_libero_90/wandb/run-*

echo "DONE"
