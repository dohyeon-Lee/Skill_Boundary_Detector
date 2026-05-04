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
#SBATCH --time=1:00:00
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
# VAE_PRIOR_SCALE=1
# MODEL=PT_libero_90_skillvla_coupled_true_embs_decoded_weight
# VAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat64_spline/spline_vae_epoch10000.pt

MODEL=PT_libero_90_skillvla_8_coupled_true_embs_decoded_8
VAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat8_spline/spline_vae_epoch10000.pt

CHECKPOINT=080000
TARGET_TASK=libero_90
USE_VAE_PRIOR=true
SAVE_SKILL_DECODE_PLOTS=true

VAE_PRIOR_POSITION_CLIP=0
ZERO_VAE_PRIOR_ORIENTATION=false
VAE_PRIOR_ORIENTATION_CLIP=0
# Leave empty to use the checkpoint config. Set to "context" or "embs" to override at eval time.
SP_PREFIX_SOURCE_OVERRIDE=""

# libero_90 전체를 한 번에 env 생성하면 EGL context가 많이 잡혀 OOM이 날 수 있음.
# 전체 평가가 필요하면 task_ids를 10개 정도씩 나눠 여러 번 돌리는 쪽이 안정적.
TASK_IDS="[0,1,2]"

WANDB_PROJECT=VLA_eval
PRIOR_TAG=$([ "${USE_VAE_PRIOR}" = "true" ] && echo "prior_on" || echo "prior_off")
if [ "${ZERO_VAE_PRIOR_ORIENTATION}" = "true" ]; then
  ORI_TAG="ori_zero"
elif awk "BEGIN {exit !(${VAE_PRIOR_ORIENTATION_CLIP} > 0)}"; then
  ORI_TAG="ori_clip${VAE_PRIOR_ORIENTATION_CLIP}"
else
  ORI_TAG="ori_prior"
fi
if awk "BEGIN {exit !(${VAE_PRIOR_POSITION_CLIP} > 0)}"; then
  POS_TAG="pos_clip${VAE_PRIOR_POSITION_CLIP}"
else
  POS_TAG="pos_prior"
fi
WANDB_RUN_NAME=${MODEL}_${CHECKPOINT}_${TARGET_TASK}_${PRIOR_TAG}_${POS_TAG}_${ORI_TAG}
POLICY_PATH=${HOMEDIR}${PROJDIR}/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi

if [ "${USE_VAE_PRIOR}" = "true" ] || [ "${SAVE_SKILL_DECODE_PLOTS}" = "true" ]; then
  VAE_PRIOR_ARG="--policy.vae_decoder_path=${VAE_CKPT}"
else
  # Override any vae_decoder_path saved in the checkpoint config.
  VAE_PRIOR_ARG="--policy.vae_decoder_path="
fi

SP_PREFIX_SOURCE_ARG=""
if [ -n "${SP_PREFIX_SOURCE_OVERRIDE}" ]; then
  SP_PREFIX_SOURCE_ARG="--policy.skill_predictor_prefix_source=${SP_PREFIX_SOURCE_OVERRIDE}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/python -m lerobot.scripts.lerobot_skillvla_eval \
  --policy.path=${POLICY_PATH} \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --env.fps=20 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  ${VAE_PRIOR_ARG} \
  --policy.use_vae_prior=${USE_VAE_PRIOR} \
  --policy.vae_prior_position_clip=${VAE_PRIOR_POSITION_CLIP} \
  --policy.zero_vae_prior_orientation=${ZERO_VAE_PRIOR_ORIENTATION} \
  --policy.vae_prior_orientation_clip=${VAE_PRIOR_ORIENTATION_CLIP} \
  ${SP_PREFIX_SOURCE_ARG} \
  --env.task_ids=${TASK_IDS} \
  --env.max_parallel_tasks=1 \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

  # --policy.vae_prior_scale=${VAE_PRIOR_SCALE} \

echo "DONE"
