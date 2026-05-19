#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_pred_label_eval

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node01
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
DATASET=libero_90_skillvla
DATASET_ROOT=libero_dataset
DATASET_SOURCE=${DATASET%_skillvla}
LABEL_DATASET_DIR=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET_SOURCE}_data/${DATASET}

LATENT_DIM=64
NUM_EMBEDDINGS=512
VQAE_STEP=10000
SKILLSET=libero_90_skillset

VQAE_CKPT=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/libero_90_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}/spline_vqae_epoch$(printf '%04d' ${VQAE_STEP}).pt

PREDICTED_LATENT_SAMPLING=true
SP_PREFIX_SOURCE=embs

MODEL=PT_${DATASET}_coupled_${PREDICTED_LATENT_SAMPLING}_${SP_PREFIX_SOURCE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}
CHECKPOINT=100000

TARGET_TASK=libero_90
TASK_IDS="[0,1,2,3,4]"
N_EPISODES=1
LABEL_EPISODE_OFFSET=0

LOAD_VQAE_DECODER=true

# Leave empty to use the checkpoint config. Set to "context" or "embs" to override at eval time.
SP_PREFIX_SOURCE_OVERRIDE=""

WANDB_PROJECT=VLA_eval
DECODER_TAG=$([ "${LOAD_VQAE_DECODER}" = "true" ] && echo "decoder_on" || echo "decoder_off")
WANDB_RUN_NAME=${MODEL}_${CHECKPOINT}_${TARGET_TASK}_pred_vs_label_${DECODER_TAG}
POLICY_PATH=${HOMEDIR}${PROJDIR}/outputs/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi

if [ ! -d "${LABEL_DATASET_DIR}" ]; then
  echo "Label dataset not found: ${LABEL_DATASET_DIR}" >&2
  exit 1
fi

if [ "${LOAD_VQAE_DECODER}" = "true" ]; then
  VQAE_DECODER_ARG="--policy.vae_decoder_path=${VQAE_CKPT}"
else
  VQAE_DECODER_ARG="--policy.vae_decoder_path="
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
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  ${VQAE_DECODER_ARG} \
  --policy.use_label_skill_tokens_eval=false \
  --policy.compare_label_skill_tokens_eval=true \
  --policy.label_skill_dataset_dir=${LABEL_DATASET_DIR} \
  --policy.label_skill_episode_offset=${LABEL_EPISODE_OFFSET} \
  ${SP_PREFIX_SOURCE_ARG} \
  --env.task_ids=${TASK_IDS} \
  --env.init_states=true \
  --env.init_state_offset=${LABEL_EPISODE_OFFSET} \
  --env.max_parallel_tasks=1 \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

echo "DONE"
