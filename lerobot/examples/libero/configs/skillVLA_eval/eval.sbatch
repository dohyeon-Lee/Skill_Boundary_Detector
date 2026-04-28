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

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL=skillVLA_libero_90_skillvla_PT
CHECKPOINT=065000
TARGET_TASK=libero_90
VAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat64_spline/spline_vae_epoch10000.pt

# libero_90 전체를 한 번에 env 생성하면 EGL context가 많이 잡혀 OOM이 날 수 있음.
# 전체 평가가 필요하면 task_ids를 10개 정도씩 나눠 여러 번 돌리는 쪽이 안정적.
TASK_IDS="[0,1,2,3,4,5,6,7,8,9]"

WANDB_PROJECT=VLA_eval
WANDB_RUN_NAME=${MODEL}_${CHECKPOINT}_${TARGET_TASK}
POLICY_PATH=${HOMEDIR}${PROJDIR}/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-eval \
  --policy.path=${POLICY_PATH} \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --env.fps=20 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.n_action_steps=1 \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --policy.vae_decoder_path=${VAE_CKPT} \
  --env.task_ids=${TASK_IDS} \
  --env.max_parallel_tasks=1 \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

echo "DONE"
