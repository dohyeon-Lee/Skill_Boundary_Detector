#!/usr/bin/env bash
#SBATCH --job-name=skill_aux
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TERMINATOR_TRAIN_CONFIG:-${SUBMIT_DIR}/terminator_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_train_config.py" \
  --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START AUXILIARY-ONLY TRAINING"
echo "host      : $(hostname)"
echo "mode      : ${TRAINING_MODE}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "terminator: enabled=${TRAIN_TERMINATOR} fsq=${FSQ_PATH} vision_frozen=${TERMINATOR_FREEZE_VISION_ENCODER}"
echo "image term: enabled=${TRAIN_IMAGE_ONLY_TERMINATOR} fresh=true vision_frozen=${IMAGE_ONLY_TERMINATOR_FREEZE_VISION_ENCODER}"
echo "wrist term: enabled=${TRAIN_WRIST_ONLY_TERMINATOR} fresh=true vision_frozen=${WRIST_ONLY_TERMINATOR_FREEZE_VISION_ENCODER}"
echo "predictor : enabled=${TRAIN_SKILL_PREDICTOR} pi_base=${PI_BASE} warm_start=${PREDICTOR_CHECKPOINT_PATH:-none}"
echo "schedule  : steps=${STEPS} mode=${SCHEDULER_MODE} warmup=${SCHEDULER_WARMUP_STEPS}"
echo "wandb     : ${WANDB_PROJECT}"
echo "output    : ${PT_OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing auxiliary dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
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

PRETRAIN_ARGS=()
if [ "${TRAIN_SKILL_PREDICTOR}" = true ]; then
  PRETRAIN_ARGS=(--policy.pretrained_path="${PI_BASE}")
fi

TRAIN_LAUNCH=("${PROJECT_ROOT}/.venv/bin/lerobot-train")
if (( NUM_GPUS > 1 )); then
  TRAIN_LAUNCH=(
    accelerate launch
    --num_processes="${NUM_GPUS}"
    "${PROJECT_ROOT}/.venv/bin/lerobot-train"
  )
  echo "launcher  : accelerate (${NUM_GPUS} processes)"
else
  echo "launcher  : direct lerobot-train (single GPU; no launcher import)"
fi

PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  "${TRAIN_LAUNCH[@]}" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_aux \
    "${PRETRAIN_ARGS[@]}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.terminator_freeze_vision_encoder="${TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}" \
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}" \
    --policy.terminator_termination_only="${TERMINATOR_TERMINATION_ONLY}" \
    --policy.train_image_only_terminator="${TRAIN_IMAGE_ONLY_TERMINATOR}" \
    --policy.image_only_terminator_freeze_vision_encoder="${IMAGE_ONLY_TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.image_only_terminator_lr_scale="${IMAGE_ONLY_TERMINATOR_LR_SCALE}" \
    --policy.image_only_terminator_end_target_sigma="${IMAGE_ONLY_TERMINATOR_END_TARGET_SIGMA}" \
    --policy.image_only_terminator_end_pos_weight="${IMAGE_ONLY_TERMINATOR_END_POS_WEIGHT}" \
    --policy.image_only_terminator_termination_only="${IMAGE_ONLY_TERMINATOR_TERMINATION_ONLY}" \
    --policy.train_wrist_only_terminator="${TRAIN_WRIST_ONLY_TERMINATOR}" \
    --policy.wrist_only_terminator_freeze_vision_encoder="${WRIST_ONLY_TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.wrist_only_terminator_lr_scale="${WRIST_ONLY_TERMINATOR_LR_SCALE}" \
    --policy.wrist_only_terminator_end_target_sigma="${WRIST_ONLY_TERMINATOR_END_TARGET_SIGMA}" \
    --policy.wrist_only_terminator_end_pos_weight="${WRIST_ONLY_TERMINATOR_END_POS_WEIGHT}" \
    --policy.wrist_only_terminator_termination_only="${WRIST_ONLY_TERMINATOR_TERMINATION_ONLY}" \
    --policy.train_skill_predictor="${TRAIN_SKILL_PREDICTOR}" \
    --policy.skill_predictor_checkpoint_path="${PREDICTOR_CHECKPOINT_PATH}" \
    --policy.skill_predictor_lr_scale="${SKILL_PREDICTOR_LR_SCALE}" \
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.skill_predictor_detach_vlm="${SKILL_PREDICTOR_DETACH_VLM}" \
    --policy.skill_predictor_lora="${SKILL_PREDICTOR_LORA}" \
    --policy.skill_predictor_lora_targets="${SKILL_PREDICTOR_LORA_TARGETS}" \
    --policy.skill_predictor_lora_rank="${SKILL_PREDICTOR_LORA_RANK}" \
    --policy.skill_predictor_lora_alpha="${SKILL_PREDICTOR_LORA_ALPHA}" \
    --policy.skill_predictor_lora_dropout="${SKILL_PREDICTOR_LORA_DROPOUT}" \
    --policy.skill_predictor_lora_lr_scale="${SKILL_PREDICTOR_LORA_LR_SCALE}" \
    --policy.skill_predictor_vlm_variant="${SKILL_PREDICTOR_VLM_VARIANT}" \
    --policy.skill_predictor_image_size="${SKILL_PREDICTOR_IMAGE_SIZE}" \
    --policy.skill_predictor_reader_tokens="${SKILL_PREDICTOR_READER_TOKENS}" \
    --policy.skill_predictor_reader_depth="${SKILL_PREDICTOR_READER_DEPTH}" \
    --policy.skill_predictor_reader_heads="${SKILL_PREDICTOR_READER_HEADS}" \
    --policy.skill_predictor_deadzone_frac="${SKILL_PREDICTOR_DEADZONE_FRAC}" \
    --policy.skill_predictor_attend_image="${SKILL_PREDICTOR_ATTEND_IMAGE}" \
    --policy.skill_predictor_attend_language="${SKILL_PREDICTOR_ATTEND_LANGUAGE}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}" \
    --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.optimizer_lr="${LR}" \
    --policy.optimizer_grad_clip_norm="${OPTIMIZER_GRAD_CLIP_NORM}" \
    --policy.scheduler_mode="${SCHEDULER_MODE}" \
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --log_freq="${LOG_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    "${RESUME_ARGS[@]}"

echo "DONE -> ${PT_OUTPUT_DIR}"
