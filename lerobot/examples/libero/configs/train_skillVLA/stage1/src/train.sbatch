#!/usr/bin/env bash
# Stage 1: unified Arch1--4 training with a fully trainable pi0.5 action expert.
#SBATCH --job-name=stage1
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
ARCHITECTURE_OVERRIDE="${STAGE1_ARCHITECTURE_OVERRIDE:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  CONFIG_ARGS+=(--architecture "${ARCHITECTURE_OVERRIDE}")
fi
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" "${CONFIG_ARGS[@]}")"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE 1"
echo "host    : $(hostname)"
echo "dataset : ${SKILLVLA_DATASET_DIR}"
echo "expert  : pi0.5 base ${PI_BASE} -> full fine-tuning"
echo "arch    : ${ARCHITECTURE_LABEL} (${ARCHITECTURE}, ${ARCHITECTURE_REVISION})"
if [ "${ARCHITECTURE}" = cond_gemma ]; then
  echo "route   : ${CONDITIONING_ROUTE}"
else
  echo "vision conditioning: ${VISION_CONDITIONING_MODE}"
  echo "Visual cross-attention queries: ${VISUAL_CROSSATTN_QUERIES}"
  echo "vision memory: ${NUM_VISUAL_LATENTS_PER_CAMERA} latents/camera"
  echo "perceiver width: ${VISUAL_PERCEIVER_WIDTH} (direct DINO/expert width)"
fi
echo "loss    : ${ACTION_LOSS_MODE}"
echo "skill   : ${TRAINING_SKILL_SOURCE}   predictor=${SKILL_PREDICTOR_CHECKPOINT_PATH:-none}"
if [ "${PHASE_BATCH_SAMPLING_ENABLED}" = true ]; then
  echo "batch   : phase-focused fraction=${PHASE_BATCH_FOCUSED_FRACTION} early_share=${PHASE_BATCH_EARLY_FRACTION} thresholds=${PHASE_BATCH_EARLY_THRESHOLD}/${PHASE_BATCH_LATE_THRESHOLD}"
else
  echo "batch   : ordinary random shuffle (architecture ablation)"
fi
echo "vision  : DINO ${DINO_MODEL_PATH}   frozen=${FREEZE_VISION_ENCODER} lr_scale=${DINO_LR_SCALE}"
echo "predict : pi0.5 VLM skill-LoRA=${SKILL_PREDICTOR_LORA} all_layers=${SKILL_PREDICTOR_ALL_LAYERS} deadzone=${SKILL_PREDICTOR_DEADZONE_FRAC}"
echo "term    : FSQ ${FSQ_PATH}   train=${TRAIN_TERMINATOR}, vision_frozen=${TERMINATOR_FREEZE_VISION_ENCODER}"
echo "schedule: steps=${STEPS}   mode=${SCHEDULER_MODE}   warmup=${SCHEDULER_WARMUP_STEPS}"
if [ "${SCHEDULER_MODE}" = cosine_decay ]; then
  echo "          lr_decay_steps=${SCHEDULER_DECAY_STEPS}"
fi
echo "output  : ${PT_OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing Stage-1 dataset: ${SKILLVLA_DATASET_DIR}" >&2
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

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_expert \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.architecture="${ARCHITECTURE}" \
    --policy.architecture_label="${ARCHITECTURE_LABEL}" \
    --policy.architecture_revision="${ARCHITECTURE_REVISION}" \
    --policy.vision_conditioning_mode="${VISION_CONDITIONING_MODE}" \
    --policy.include_state_in_visual_crossattn="${INCLUDE_STATE_IN_VISUAL_CROSSATTN}" \
    --policy.include_skill_in_visual_crossattn="${INCLUDE_SKILL_IN_VISUAL_CROSSATTN}" \
    --policy.num_visual_latents_per_camera="${NUM_VISUAL_LATENTS_PER_CAMERA}" \
    --policy.visual_perceiver_width="${VISUAL_PERCEIVER_WIDTH}" \
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.cond_encoder_variant="${COND_ENCODER_VARIANT}" \
    --policy.conditioning_route="${CONDITIONING_ROUTE}" \
    --policy.vision_backbone=dino \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.dino_image_size="${DINO_IMAGE_SIZE}" \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.dino_lr_scale="${DINO_LR_SCALE}" \
    --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.transition_jitter_pmax="${TRANSITION_JITTER_PMAX}" \
    --policy.transition_jitter_distribution="${TRANSITION_JITTER_DISTRIBUTION}" \
    --policy.phase_batch_sampling_enabled="${PHASE_BATCH_SAMPLING_ENABLED}" \
    --policy.phase_batch_focused_fraction="${PHASE_BATCH_FOCUSED_FRACTION}" \
    --policy.phase_batch_early_fraction="${PHASE_BATCH_EARLY_FRACTION}" \
    --policy.phase_batch_early_threshold="${PHASE_BATCH_EARLY_THRESHOLD}" \
    --policy.phase_batch_late_threshold="${PHASE_BATCH_LATE_THRESHOLD}" \
    --policy.training_skill_source="${TRAINING_SKILL_SOURCE}" \
    --policy.skill_predictor_checkpoint_path="${SKILL_PREDICTOR_CHECKPOINT_PATH}" \
    --policy.train_skill_predictor="${TRAIN_SKILL_PREDICTOR}" \
    --policy.skill_predictor_weight="${SKILL_PREDICTOR_WEIGHT}" \
    --policy.skill_predictor_lr_scale="${SKILL_PREDICTOR_LR_SCALE}" \
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.skill_predictor_detach_vlm="${SKILL_PREDICTOR_DETACH_VLM}" \
    --policy.skill_predictor_lora="${SKILL_PREDICTOR_LORA}" \
    --policy.skill_predictor_lora_targets="${SKILL_PREDICTOR_LORA_TARGETS}" \
    --policy.skill_predictor_lora_rank="${SKILL_PREDICTOR_LORA_RANK}" \
    --policy.skill_predictor_lora_alpha="${SKILL_PREDICTOR_LORA_ALPHA}" \
    --policy.skill_predictor_lora_dropout="${SKILL_PREDICTOR_LORA_DROPOUT}" \
    --policy.skill_predictor_lora_lr_scale="${SKILL_PREDICTOR_LORA_LR_SCALE}" \
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
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}" \
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.action_loss_mode="${ACTION_LOSS_MODE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.min_period="${MIN_PERIOD}" \
    --policy.max_period="${MAX_PERIOD}" \
    --policy.time_sampling_beta_alpha="${TIME_SAMPLING_BETA_ALPHA}" \
    --policy.time_sampling_beta_beta="${TIME_SAMPLING_BETA_BETA}" \
    --policy.time_sampling_scale="${TIME_SAMPLING_SCALE}" \
    --policy.time_sampling_offset="${TIME_SAMPLING_OFFSET}" \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.vsa_debug_steps="${VSA_DEBUG_STEPS}" \
    --policy.vsa_debug_schedule="${VSA_DEBUG_SCHEDULE}" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.compile_model=false \
    --policy.optimizer_lr="${LR}" \
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
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
