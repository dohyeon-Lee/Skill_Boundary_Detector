#!/usr/bin/env bash
#SBATCH --job-name=DP_train_90
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

export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

N_OBS_STEPS=5    # obs history 길이
N_ACTION_STEPS=16  # action chunk 길이
HORIZON=16         # UNet prediction horizon

TRAIN_CONFIG_PATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/examples/libero/configs/DP_train/train_dp_libero90.yaml
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero_90_obs${N_OBS_STEPS}_horizon${HORIZON}
WANDB_PROJECT=DP_train
JOB_NAME=DP_libero90_obs${N_OBS_STEPS}_horizon${HORIZON}
SAVE_FREQ=2000   # 체크포인트 저장 주기 (steps)

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
  --job_name=${JOB_NAME} \
  --wandb.enable=true \
  --wandb.project=${WANDB_PROJECT} \
  --save_freq=${SAVE_FREQ} \
  --policy.n_obs_steps=${N_OBS_STEPS} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.horizon=${HORIZON} \
  ${RESUME_FLAG}

echo "DONE"
