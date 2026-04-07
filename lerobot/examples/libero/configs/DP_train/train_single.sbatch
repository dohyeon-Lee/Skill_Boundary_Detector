#!/usr/bin/env bash
#SBATCH --job-name=DP_train_SO
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot

source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

TRAIN_CONFIG_PATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/examples/libero/configs/DP_train/train_dp_libero_spatial_object.yaml
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero_spatial_object

RESUME_FLAG=""
CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  lerobot-train \
  --config_path=${TRAIN_CONFIG_PATH} \
  --output_dir=${OUTPUT_DIR} \
  ${RESUME_FLAG}

echo "DONE"
