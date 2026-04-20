#!/usr/bin/env bash
#SBATCH --job-name=pi05_eval
#SBATCH --partition=asus_pro6000,suma_a6000,asus_6000ada,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# MODEL=pi05_libero_90
# CHECKPOINT=100000
# TARGET_TASK=libero_90

MODEL=pi05_libero_spatial_object
CHECKPOINT=046000
TARGET_TASK=libero_spatial

WANDB_PROJECT=VLA_eval
WANDB_RUN_NAME=${MODEL}PT_${CHECKPOINT}_${TARGET_TASK}


PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/lerobot-eval \
  --policy.path=/scratch/mdorazi/Skill_Boundary_Detector/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --env.task_ids="[0,1,2]" \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'
