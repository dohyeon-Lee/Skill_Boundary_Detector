#!/usr/bin/env bash
#SBATCH --job-name=DP_train

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

N_OBS_STEPS=10    # obs history 길이
N_ACTION_STEPS=16  # action chunk 길이
HORIZON=16         # UNet prediction horizon
DATASET=libero_90

TRAIN_CONFIG_PATH=${HOMEDIR}${PROJDIR}/lerobot/examples/libero/configs/DP_train/train_dp_${DATASET}.yaml
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/dp_${DATASET}_obs${N_OBS_STEPS}_horizon${HORIZON}
WANDB_PROJECT=DP_train
JOB_NAME=DP_${DATASET}_obs${N_OBS_STEPS}_horizon${HORIZON}
SAVE_FREQ=5000   # 체크포인트 저장 주기 (steps)

RESUME_FLAG=""
CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
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
  --dataset.repo_id=lerobot/${DATASET} \
  --dataset.root=${HOMEDIR}${PROJDIR}/libero_dataset/${DATASET} \
  ${RESUME_FLAG}

echo "DONE"
