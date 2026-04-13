#!/usr/bin/env bash
#SBATCH --job-name=pi05_skill_train
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000
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
    --dataset.repo_id=mdorazi/libero_SO_skill \
    --dataset.root=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_spatial_object_skill \
    --policy.type=pi05 \
    --output_dir=../outputs/pi05_skill_SO \
    --job_name=pi05_skill_training_SO \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_skill_generator=true \
    --policy.skill_z_dim=8 \
    --policy.skill_lstm_hidden_dim=512 \
    --policy.skill_lstm_num_layers=2 \
    --policy.skill_loss_weight_i=1.0 \
    --policy.skill_loss_weight_z=1.0 \
    --policy.skill_teacher_forcing=true \
    --policy.push_to_hub=false \
    --save_freq=1000 \
    --steps=30000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=8

echo "DONE"
