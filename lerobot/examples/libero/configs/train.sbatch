#!/usr/bin/env bash
#SBATCH --job-name=DP_train
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos #big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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

# Parameter combinations: horizon=16, n_action_steps=12/16, n_obs_steps=5/10
N_OBS_STEPS=( 5  5 10 10)
HORIZON=(    16 16 16 16)
N_ACTION=(   8 16 8 16)

IDX=$SLURM_ARRAY_TASK_ID
OBS=${N_OBS_STEPS[$IDX]}
HOR=${HORIZON[$IDX]}
ACT=${N_ACTION[$IDX]}

EXP_NAME="dp_libero90_obs${OBS}_hor${HOR}_act${ACT}"
OUTPUT_DIR="/scratch/mdorazi/Skill_Boundary_Detector/outputs/${EXP_NAME}"

echo "Config: n_obs_steps=${OBS}, horizon=${HOR}, n_action_steps=${ACT}"
echo "Output: ${OUTPUT_DIR}"

TRAIN_CONFIG_PATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/examples/libero/configs/train_dp_libero90_rawteleop.yaml
RESUME_FLAG=""
CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  # 체크포인트 있으면 resume
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  # output dir만 있고 체크포인트 없으면 삭제 후 재시작
  rm -rf "${OUTPUT_DIR}"
fi

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  lerobot-train \
  --config_path=${TRAIN_CONFIG_PATH} \
  --policy.n_obs_steps=${OBS} \
  --policy.horizon=${HOR} \
  --policy.n_action_steps=${ACT} \
  --output_dir=${OUTPUT_DIR} \
  ${RESUME_FLAG}

# sbatch --array=3 train.sbatch
