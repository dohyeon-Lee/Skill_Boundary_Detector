#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_eval_vqvae

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

# Avoid system cuDNN shadowing the cuDNN bundled with the installed PyTorch.
unset LD_LIBRARY_PATH

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── SkillVLA checkpoint ──────────────────────────────────────────────────────
LATENT_DIM=128
NUM_EMBEDDINGS=128

# Set this directly so PT and FT checkpoints can be evaluated without changing
# dataset/label variables below.
# Example PT: PT_libero_90_skillvla_true_embs_lat128_vq128
# Example FT: FT_025000PT_libero_10_op1_10_skillvlaFT_true_embs_lat128_vq128
MODEL=FT_025000PT_libero_10_op1_10_skillvlaFT_true_embs_lat128_vq128_true # PT_libero_90_skillvla_true_embs_lat128_vq128
CHECKPOINT=030000 # 025000
POLICY_PATH=${HOMEDIR}${PROJDIR}/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model

# ── VQ-VAE decoder used for end-signal inference ─────────────────────────────
SKILLSET=libero_90_skillset 
VQVAE_STEP=1500
VQVAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/libero_90_vqvae_dinov3_vits16_gru_mlp_zero_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_0.1_exp3/spline_vqae_epoch$(printf '%04d' ${VQVAE_STEP}).pt

SKILL_DECODER_END_THRESHOLD=0.5
SKILL_DECODER_STATE_INDICES="[0,1,2,3,4,5]"
INFERENCE_SKILL_MAX_ORDER=20
INFERENCE_SKILL_MAX_LENGTH=200

# ── Eval settings ────────────────────────────────────────────────────────────
TARGET_TASK=libero_10
TASK_IDS="[0,1,2,3,4,5,6,7,8,9]"
N_EPISODES=1
N_ACTION_STEPS=5

# Optional oracle/diagnostic modes. Default eval uses predicted skill tokens.
USE_LABEL_SKILL_TOKENS=false  # false
COMPARE_LABEL_SKILL_TOKENS=true
LABEL_EPISODE_OFFSET=0

# Keep DATASET/DATASET_ROOT tied to POLICY_PATH above. Label tokens can come from
# a different add_skill_latents_to_dataset.py output, e.g. libero_10_skillvla.
LABEL_DATASET=libero_10_op1_10_skillvla # libero_10_op1_10_skillvla
LABEL_DATASET_ROOT=libero_dataset
LABEL_DATASET_SOURCE=${LABEL_DATASET%_skillvla}
LABEL_DATASET_DIR=${HOMEDIR}${PROJDIR}/${LABEL_DATASET_ROOT}/${LABEL_DATASET_SOURCE}_data/${LABEL_DATASET}

# Leave empty to use the checkpoint config. Set to "context" or "embs" to override.
SP_PREFIX_SOURCE_OVERRIDE=""

WANDB_PROJECT=VLA_eval
LABEL_TAG=$([ "${USE_LABEL_SKILL_TOKENS}" = "true" ] && echo "label_tokens" || echo "pred_tokens")
COMPARE_TAG=$([ "${COMPARE_LABEL_SKILL_TOKENS}" = "true" ] && echo "_compare" || echo "")
WANDB_RUN_NAME=${MODEL}_${CHECKPOINT}_${TARGET_TASK}_${LABEL_TAG}${COMPARE_TAG}_vqvae_end

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi

if [ ! -f "${VQVAE_CKPT}" ]; then
  echo "VQ-VAE checkpoint not found: ${VQVAE_CKPT}" >&2
  exit 1
fi

LABEL_ARGS=""
if [ "${USE_LABEL_SKILL_TOKENS}" = "true" ] || [ "${COMPARE_LABEL_SKILL_TOKENS}" = "true" ]; then
  if [ ! -d "${LABEL_DATASET_DIR}" ]; then
    echo "Label dataset not found: ${LABEL_DATASET_DIR}" >&2
    exit 1
  fi
  LABEL_ARGS="--policy.label_skill_dataset_dir=${LABEL_DATASET_DIR} --policy.label_skill_episode_offset=${LABEL_EPISODE_OFFSET}"
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
  --eval.n_episodes=${N_EPISODES} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --policy.vae_decoder_path=${VQVAE_CKPT} \
  --policy.freeze_vae_decoder=true \
  --policy.skill_latent_dim=${LATENT_DIM} \
  --policy.skill_predictor_num_embeddings=${NUM_EMBEDDINGS} \
  --policy.skill_decoder_end_threshold=${SKILL_DECODER_END_THRESHOLD} \
  --policy.skill_decoder_state_indices=${SKILL_DECODER_STATE_INDICES} \
  --policy.inference_skill_max_order=${INFERENCE_SKILL_MAX_ORDER} \
  --policy.inference_skill_max_length=${INFERENCE_SKILL_MAX_LENGTH} \
  --policy.use_label_skill_tokens_eval=${USE_LABEL_SKILL_TOKENS} \
  --policy.compare_label_skill_tokens_eval=${COMPARE_LABEL_SKILL_TOKENS} \
  ${LABEL_ARGS} \
  ${SP_PREFIX_SOURCE_ARG} \
  --env.task_ids=${TASK_IDS} \
  --env.max_parallel_tasks=1 \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

echo "DONE"
