#!/usr/bin/env bash
#SBATCH --job-name=vae_train
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

EXP_NAME="vae90_lat8_hid128"
OUTPUT_DIR="/scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_90/${EXP_NAME}"

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/train_vae.py \
    --skills_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero90 \
    --output_dir ${OUTPUT_DIR} \
    --latent_dim 8 \
    --hidden_dim 128 \
    --num_layers 2 \
    --epochs 500 \
    --beta 0.01 \
    --lr 3e-4 \
    --batch_size 32 \
    --min_skill_len 10 \
    --checkpoint_every 100 \
    --wandb_project SBD_vae_90 \
    --wandb_run_name ${EXP_NAME}

echo "DONE ${EXP_NAME}"
