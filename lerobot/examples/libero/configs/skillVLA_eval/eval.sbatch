#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_eval

# SBATCH --partition=asus_pro6000
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

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL=skillVLA_libero_90_skillvla_PT
CHECKPOINT=060000
TARGET_TASK=libero_90

WANDB_PROJECT=VLA_eval
WANDB_RUN_NAME=${MODEL}_${CHECKPOINT}_${TARGET_TASK}

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-eval \
  --policy.path=${HOMEDIR}${PROJDIR}/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --env.task_ids="[0,1,2,3,4,5,6,7,8,9]" \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

echo "DONE"
