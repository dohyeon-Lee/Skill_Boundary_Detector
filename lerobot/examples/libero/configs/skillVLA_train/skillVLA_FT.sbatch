#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_FT

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

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
    CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
fi
if [ ! -f "${CONFIG_PY}" ]; then
    echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
    echo "Submit from lerobot root or from examples/libero/configs/skillVLA_train." >&2
    exit 1
fi
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
    BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
    BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell)"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── Fine-tune dataset / base checkpoint ─────────────────────────────────────
# All generated artifacts live under:
#   libero_dataset/${DATA}_data/
FT_DATA=${FT_DATA:?FT_DATA must be set by pipeline_config.yaml or the environment}
PT_DATA=${PT_DATA:-${FT_PT_DATA}}
eval "$(python "${CONFIG_PY}" --shell --data "${FT_DATA}" --prefix FT_)"
eval "$(python "${CONFIG_PY}" --shell --data "${PT_DATA}" --prefix PT_)"
FT_DATASET=${FT_SKILLVLA_DATASET}
FT_DATASET_ROOT=${FT_DATADIR}
FT_DATASET_DIR=${FT_SKILLVLA_DATASET_DIR}
FT_DINO_TOKENS_PATH=${FT_DINO_TOKENS_PATH}
FT_DINO_CACHE_PATH=${FT_DINO_CACHE_PATH}

# SkillVLA checkpoint trained on libero_90.
VLA_OUTPUT_ROOT=${HOMEDIR}${PROJDIR}/VLA_outputs
PT_BATCH_SIZE=${PT_BATCH_SIZE:-${FT_PT_BATCH_SIZE}}
PT_EXP=${PT_EXP:-${FT_PT_EXP}}
PT_FSQ_TAG=fsq${PT_FSQ_LEVELS_STR// /}
if [ "${FT_PT_BLOCK_LANG}" = "true" ]; then
    PT_BLOCK_LANG_TAG=blocklang
else
    PT_BLOCK_LANG_TAG=allowlang
fi
if [ -n "${FT_PT_MODEL}" ]; then
    PT_RUN_NAME=${FT_PT_MODEL}
else
    PT_RUN_NAME=PT_${PT_SKILLVLA_DATASET}_${PT_FSQ_TAG}_FSQ_epoch${PT_FSQ_EPOCH}_${PT_BLOCK_LANG_TAG}_batch${PT_BATCH_SIZE}
    if [ -n "${PT_EXP}" ]; then
        PT_RUN_NAME=${PT_RUN_NAME}_${PT_EXP}
    fi
fi
PT_CHECKPOINT=${PT_CHECKPOINT:-${FT_PT_CHECKPOINT}}
BASE_POLICY=${VLA_OUTPUT_ROOT}/${PT_RUN_NAME}/checkpoints/${PT_CHECKPOINT}/pretrained_model

# FSQ decoder used by the base SkillVLA run. Keep this matched to PT.
FSQ_LEVELS_ARG=${PT_FSQ_LEVELS_ARG}
FSQ_DIM=${PT_FSQ_DIM}
FSQ_NUM_EMBEDDINGS=${PT_FSQ_NUM_EMBEDDINGS}
FSQ_RUN=${PT_FSQ_RUN}
FSQ_CKPT=${PT_FSQ_CKPT}
FSQ_IMAGE_MODEL_PATH=${PT_IMAGE_MODEL_PATH}

if [ ! -f "${BASE_POLICY}/model.safetensors" ]; then
    echo "Missing base SkillVLA checkpoint: ${BASE_POLICY}"
    exit 1
fi

if [ ! -d "${FT_DATASET_DIR}" ]; then
    echo "Missing fine-tune SkillVLA dataset: ${FT_DATASET_DIR}"
    exit 1
fi

if [ ! -f "${FSQ_CKPT}" ]; then
    echo "Missing FSQ checkpoint: ${FSQ_CKPT}"
    exit 1
fi

if [ ! -f "${FT_DINO_TOKENS_PATH}" ]; then
    echo "Missing fine-tune DINO token npz: ${FT_DINO_TOKENS_PATH}"
    exit 1
fi

# ── Optimization ─────────────────────────────────────────────────────────────
LR=2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4
NUM_GPUS=1
EXP=${EXP:-${FT_EXP}}

# ── SkillVLA losses / boundary randomization ────────────────────────────────
SP_LOSS_WEIGHT=1.0
SKILL_DECODER_LOSS_WEIGHT=1.0
SKILL_DECODER_DELTA_LOSS_WEIGHT=10.0
SKILL_DECODER_END_LOSS_WEIGHT=1.0
SKILL_DECODER_END_POS_WEIGHT=1.0
SKILL_DECODER_END_THRESHOLD=0.5
SKILL_DECODER_STATE_INDICES=${PT_SKILL_DECODER_STATE_INDICES}
SKILL_DECODER_PRIOR_NOISE_RATIO=${FT_SKILL_DECODER_PRIOR_NOISE_RATIO}
FREEZE_VAE_DECODER=true
INFERENCE_SKILL_MAX_LENGTH=${PT_INFERENCE_SKILL_MAX_LENGTH}
SKILL_BOUNDARY_RANDOM_P=10

# Fine-tuning follows pi05_FT: keep the vision encoder frozen by default.
FREEZE_VISION_ENCODER=${FT_FREEZE_VISION_ENCODER}
# true: freeze the full Paligemma VLM, while leaving the action expert,
# skill predictor, and SkillVLA projection/decoder modules controlled by
# their own freeze flags.
TRAIN_EXPERT_ONLY=${FT_TRAIN_EXPERT_ONLY}

# BLOCK_LANG=true이면 action expert의 language-token attention을 막습니다.
# false이면 action expert가 language + image/state prefix 전체를 볼 수 있습니다.
# pipeline_config.yaml 의 block_lang 값에서 옵니다.
# DETACH_ACTION_PREFIX_GRAD=true이면 action flow loss는 prefix/VLM 경로를 업데이트하지 않습니다.

RUN_NAME=${FT_DATASET}_${PT_FSQ_TAG}_${PT_ACTION_MODE}_${PT_DECODER_IMAGE_MODE}_FSQ_epoch${PT_FSQ_EPOCH}
if [ -n "${EXP}" ]; then
    RUN_NAME=${RUN_NAME}_${EXP}
fi
OUTPUT_DIR=${VLA_OUTPUT_ROOT}/FT_${RUN_NAME}
CONFIG_PATH=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json

RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint found. Removing existing output dir and starting fresh."
    rm -rf "${OUTPUT_DIR}"
else
    echo "No checkpoint found. Starting fresh fine-tuning."
fi

# ── SkillVLA fine-tuning from libero_90 checkpoint ──────────────────────────
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} \
    ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=dohyeon/${FT_DATASET} \
    --dataset.root=${FT_DATASET_DIR} \
    --policy.type=skill_vla \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --policy.pretrained_path=${BASE_POLICY} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_Finetune \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=${FREEZE_VISION_ENCODER} \
    --policy.train_expert_only=${TRAIN_EXPERT_ONLY} \
    --policy.push_to_hub=false \
    --policy.block_lang_to_action=${FT_BLOCK_LANG} \
    --policy.detach_action_prefix_grad=${FT_DETACH_ACTION_PREFIX_GRAD} \
    --policy.vae_decoder_path=${FSQ_CKPT} \
    --policy.skill_decoder_image_model_name=${FSQ_IMAGE_MODEL_PATH} \
    --policy.freeze_vae_decoder=${FREEZE_VAE_DECODER} \
    --policy.skill_latent_dim=${FSQ_DIM} \
    --policy.skill_fsq_levels=${FSQ_LEVELS_ARG} \
    --policy.skill_predictor_num_embeddings=${FSQ_NUM_EMBEDDINGS} \
    --policy.skill_decoder_dino_tokens_path=${FT_DINO_TOKENS_PATH} \
    --policy.skill_decoder_dino_cache_path=${FT_DINO_CACHE_PATH} \
    --policy.skill_decoder_dino_build_cache=true \
    --policy.skill_predictor_loss_weight=${SP_LOSS_WEIGHT} \
    --policy.skill_decoder_loss_weight=${SKILL_DECODER_LOSS_WEIGHT} \
    --policy.skill_decoder_delta_loss_weight=${SKILL_DECODER_DELTA_LOSS_WEIGHT} \
    --policy.skill_decoder_end_loss_weight=${SKILL_DECODER_END_LOSS_WEIGHT} \
    --policy.skill_decoder_end_pos_weight=${SKILL_DECODER_END_POS_WEIGHT} \
    --policy.skill_decoder_end_threshold=${SKILL_DECODER_END_THRESHOLD} \
    --policy.skill_decoder_state_indices=${SKILL_DECODER_STATE_INDICES} \
    --policy.skill_decoder_prior_noise_ratio=${SKILL_DECODER_PRIOR_NOISE_RATIO} \
    --policy.inference_skill_max_length=${INFERENCE_SKILL_MAX_LENGTH} \
    --policy.skill_boundary_random_p=${SKILL_BOUNDARY_RANDOM_P} \
    --save_freq=2000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}

echo "DONE"
