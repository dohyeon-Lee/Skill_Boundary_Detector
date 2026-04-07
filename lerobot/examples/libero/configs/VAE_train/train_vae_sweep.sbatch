#!/usr/bin/env bash
#SBATCH --job-name=vae_sweep
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot
source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# ── Experiment grid ────────────────────────────────────────────────────────────
#         latent_dim  hidden_dim
LATENT=(  32          64          32          64  )
HIDDEN=(  128         128         256         256 )

IDX=$SLURM_ARRAY_TASK_ID
LAT=${LATENT[$IDX]}
HID=${HIDDEN[$IDX]}

EXP_NAME="vae_lat${LAT}_hid${HID}"
OUTPUT_DIR="/scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_sweep_epoch/${EXP_NAME}"

echo "Config: latent=${LAT}, hidden=${HID}"
echo "Output: ${OUTPUT_DIR}"

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/train_vae.py \
    --skills_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero10 \
    --output_dir ${OUTPUT_DIR} \
    --latent_dim ${LAT} \
    --hidden_dim ${HID} \
    --num_layers 2 \
    --epochs 1000 \
    --beta 0.01 \
    --lr 3e-4 \
    --batch_size 32 \
    --min_skill_len 10 \
    --checkpoint_every 200 \
    --wandb_project SBD_vae \
    --wandb_run_name ${EXP_NAME}

echo "DONE ${EXP_NAME}"
