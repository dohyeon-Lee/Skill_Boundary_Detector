#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_fsq_train

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

CONFIG_PY=examples/libero/configs/data_generation/pipeline_config.py
eval "$(python "${CONFIG_PY}" --shell)"

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── Dataset / checkpoints ────────────────────────────────────────────────────
# All generated artifacts live under:
#   libero_dataset/${DATA}_data/
DATASET=${SKILLVLA_DATASET}
DATASET_ROOT=${DATADIR}
DATASET_DIR=${SKILLVLA_DATASET_DIR}
PI_BASE=lerobot/pi05_base

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Missing SkillVLA dataset: ${DATASET_DIR}"
    exit 1
fi

if [ ! -f "${FSQ_CKPT}" ]; then
    echo "Missing FSQ checkpoint: ${FSQ_CKPT}"
    exit 1
fi

if [ ! -f "${DINO_TOKENS_PATH}" ]; then
    echo "Missing DINO token npz: ${DINO_TOKENS_PATH}"
    exit 1
fi

# ── Optimization ─────────────────────────────────────────────────────────────
LR=2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4
NUM_GPUS=1

# ── SkillVLA losses / boundary randomization ────────────────────────────────
SP_LOSS_WEIGHT=1.0
SKILL_DECODER_LOSS_WEIGHT=1.0
SKILL_DECODER_END_POS_WEIGHT=1.0
SKILL_DECODER_END_THRESHOLD=0.5
SKILL_DECODER_STATE_INDICES=${SKILL_DECODER_STATE_INDICES}
FREEZE_VAE_DECODER=false
INFERENCE_SKILL_MAX_ORDER=${INFERENCE_SKILL_MAX_ORDER}
INFERENCE_SKILL_MAX_LENGTH=${INFERENCE_SKILL_MAX_LENGTH}
SKILL_BOUNDARY_RANDOM_P=10

# true: action expert attends to image/state prefix tokens, not language tokens.
# pipeline_config.yaml 의 block_lang 값에서 옵니다.

# context: Paligemma prefix-only forward 이후 contextual hidden
# embs: embed_prefix 직후 raw prefix token embedding
SP_PREFIX_SOURCE=embs

# Scheduled sampling: linearly ramp from GT token → predicted token
PREDICTED_LATENT_SAMPLING=true
PREDICTED_LATENT_START_STEP=0
PREDICTED_LATENT_RAMP_STEPS=50000
PREDICTED_LATENT_MAX_PROB=1.0

FSQ_RUN_TAG=${FSQ_RUN#${FSQ_MODEL_DATA}_}
RUN_NAME=${DATASET}_${FSQ_RUN_TAG}_${PREDICTED_LATENT_SAMPLING}_${SP_PREFIX_SOURCE}
OUTPUT_DIR=../outputs/PT_${RUN_NAME}
CONFIG_PATH=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json

RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint found. Removing existing output dir and starting fresh."
    rm -rf "${OUTPUT_DIR}"
else
    echo "No checkpoint found. Starting fresh training."
fi

# ── SkillVLA FSQ-end training ────────────────────────────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${DATASET} \
    --dataset.root=${DATASET_DIR} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --policy.pretrained_path=${PI_BASE} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.block_lang_to_action=${BLOCK_LANG} \
    --policy.vae_decoder_path=${FSQ_CKPT} \
    --policy.freeze_vae_decoder=${FREEZE_VAE_DECODER} \
    --policy.skill_latent_dim=${FSQ_DIM} \
    --policy.skill_fsq_levels=${FSQ_LEVELS_ARG} \
    --policy.skill_predictor_num_embeddings=${FSQ_NUM_EMBEDDINGS} \
    --policy.skill_decoder_dino_tokens_path=${DINO_TOKENS_PATH} \
    --policy.skill_decoder_dino_cache_path=${DINO_CACHE_PATH} \
    --policy.skill_decoder_dino_build_cache=true \
    --policy.skill_predictor_prefix_source=${SP_PREFIX_SOURCE} \
    --policy.skill_predictor_loss_weight=${SP_LOSS_WEIGHT} \
    --policy.skill_decoder_loss_weight=${SKILL_DECODER_LOSS_WEIGHT} \
    --policy.skill_decoder_end_pos_weight=${SKILL_DECODER_END_POS_WEIGHT} \
    --policy.skill_decoder_end_threshold=${SKILL_DECODER_END_THRESHOLD} \
    --policy.skill_decoder_state_indices=${SKILL_DECODER_STATE_INDICES} \
    --policy.inference_skill_max_order=${INFERENCE_SKILL_MAX_ORDER} \
    --policy.inference_skill_max_length=${INFERENCE_SKILL_MAX_LENGTH} \
    --policy.skill_boundary_random_p=${SKILL_BOUNDARY_RANDOM_P} \
    --policy.predicted_latent_sampling=${PREDICTED_LATENT_SAMPLING} \
    --policy.predicted_latent_start_step=${PREDICTED_LATENT_START_STEP} \
    --policy.predicted_latent_ramp_steps=${PREDICTED_LATENT_RAMP_STEPS} \
    --policy.predicted_latent_max_prob=${PREDICTED_LATENT_MAX_PROB} \
    --save_freq=5000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}


echo "DONE"
