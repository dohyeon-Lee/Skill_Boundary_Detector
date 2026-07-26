#!/usr/bin/env bash
# Stage 1: language-free VSA pretraining with a fully trainable pi0.5 action expert.
#SBATCH --job-name=stage1_vsa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SUBMIT_DIR}/stage1_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE 1 VSA"
echo "dataset : ${SKILLVLA_DATASET_DIR}"
echo "expert  : pi0.5 base ${PI_BASE} -> full fine-tuning"
echo "vision  : DINO ${DINO_MODEL_PATH}   frozen=${FREEZE_VISION_ENCODER}"
echo "predict : frozen pi0.5 VLM -> reader/head   train=${TRAIN_SKILL_PREDICTOR}"
echo "term    : FSQ ${FSQ_PATH}   train=${TRAIN_TERMINATOR}, vision_frozen=${TERMINATOR_FREEZE_VISION_ENCODER}"
echo "output  : ${PT_OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing Stage-1 dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${DINO_LR}" ]; then
  EXTRA_ARGS+=(--policy.dino_lr="${DINO_LR}")
fi

CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found; resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -e "${PT_OUTPUT_DIR}" ]; then
  echo "Output exists without a resumable checkpoint: ${PT_OUTPUT_DIR}" >&2
  exit 1
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_expert \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.cond_encoder_variant="${COND_ENCODER_VARIANT}" \
    --policy.vision_backbone=dino \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.dino_image_size="${DINO_IMAGE_SIZE}" \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.state_cond_mode="${STATE_COND_MODE}" \
    --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.transition_jitter_pmax="${TRANSITION_JITTER_PMAX}" \
    --policy.transition_jitter_distribution="${TRANSITION_JITTER_DISTRIBUTION}" \
    --policy.train_skill_predictor="${TRAIN_SKILL_PREDICTOR}" \
    --policy.skill_predictor_weight="${SKILL_PREDICTOR_WEIGHT}" \
    --policy.skill_predictor_lr_scale="${SKILL_PREDICTOR_LR_SCALE}" \
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.skill_predictor_detach_vlm=true \
    --policy.skill_predictor_reader_tokens="${SKILL_PREDICTOR_READER_TOKENS}" \
    --policy.skill_predictor_reader_depth="${SKILL_PREDICTOR_READER_DEPTH}" \
    --policy.skill_predictor_reader_heads="${SKILL_PREDICTOR_READER_HEADS}" \
    --policy.skill_predictor_deadzone_frac="${SKILL_PREDICTOR_DEADZONE_FRAC}" \
    --policy.skill_predictor_attend_image="${SKILL_PREDICTOR_ATTEND_IMAGE}" \
    --policy.skill_predictor_attend_language="${SKILL_PREDICTOR_ATTEND_LANGUAGE}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.terminator_freeze_vision_encoder="${TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.terminator_dino_model_path="${TERMINATOR_DINO_MODEL_PATH}" \
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}" \
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.min_period="${MIN_PERIOD}" \
    --policy.max_period="${MAX_PERIOD}" \
    --policy.time_sampling_beta_alpha="${TIME_SAMPLING_BETA_ALPHA}" \
    --policy.time_sampling_beta_beta="${TIME_SAMPLING_BETA_BETA}" \
    --policy.time_sampling_scale="${TIME_SAMPLING_SCALE}" \
    --policy.time_sampling_offset="${TIME_SAMPLING_OFFSET}" \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.compile_model=false \
    --policy.optimizer_lr="${LR}" \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --log_freq="${LOG_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
