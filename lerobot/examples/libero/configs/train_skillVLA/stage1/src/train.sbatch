#!/usr/bin/env bash
# Stage 1: unified Arch1--4 training with a fully trainable pi0.5 action expert.
#SBATCH --job-name=stage1
#SBATCH --gres=gpu:1
#SBATCH --requeue
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

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
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
echo "base loss: flow (fixed)"
echo "loss mask: after_skill_end=${MASK_ACTIONS_AFTER_SKILL_END}"
echo "proprio : grounding=${PROPRIO_GROUNDING}"
echo "cumulative XYZ: enabled=${CUMULATIVE_XYZ_LOSS_ENABLED} weight=${CUMULATIVE_XYZ_LOSS_WEIGHT}"
echo "mode z  : enabled=${SKILL_FLOW_LATENT_BEST_OF_N_ENABLED} N=${SKILL_FLOW_LATENT_CANDIDATES} K=${SKILL_FLOW_LATENT_TOP_K} M=${SKILL_FLOW_LATENT_ASSIGNMENT_TIMESTEPS} ranking=${SKILL_FLOW_LATENT_RANKING_ROUTE} fp32=${SKILL_FLOW_LATENT_FP32}"
echo "skill   : ${TRAINING_SKILL_SOURCE}   predictor=${SKILL_PREDICTOR_CHECKPOINT_PATH:-none}"
echo "batch   : ordinary random shuffle"
echo "vision  : DINO ${DINO_MODEL_PATH}   frozen=${FREEZE_VISION_ENCODER} lr_scale=${DINO_LR_SCALE}"
echo "predict : frozen skill source=${SKILL_PREDICTOR_CHECKPOINT_PATH:-none}"
echo "train   : action model only (no predictor/terminator co-training)"
echo "optim   : muon=${USE_MUON} base_lr=${LR}"
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
    --policy.transition_jitter_early_start_pmax="${TRANSITION_JITTER_EARLY_START_PMAX}" \
    --policy.transition_jitter_late_start_pmax="${TRANSITION_JITTER_LATE_START_PMAX}" \
    --policy.transition_jitter_early_end_pmax="${TRANSITION_JITTER_EARLY_END_PMAX}" \
    --policy.transition_jitter_late_end_pmax="${TRANSITION_JITTER_LATE_END_PMAX}" \
    --policy.transition_jitter_distribution="${TRANSITION_JITTER_DISTRIBUTION}" \
    --policy.training_skill_source="${TRAINING_SKILL_SOURCE}" \
    --policy.skill_predictor_checkpoint_path="${SKILL_PREDICTOR_CHECKPOINT_PATH}" \
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.skill_predictor_detach_vlm="${SKILL_PREDICTOR_DETACH_VLM}" \
    --policy.skill_predictor_lora="${SKILL_PREDICTOR_LORA}" \
    --policy.skill_predictor_lora_targets="${SKILL_PREDICTOR_LORA_TARGETS}" \
    --policy.skill_predictor_lora_rank="${SKILL_PREDICTOR_LORA_RANK}" \
    --policy.skill_predictor_lora_alpha="${SKILL_PREDICTOR_LORA_ALPHA}" \
    --policy.skill_predictor_lora_dropout="${SKILL_PREDICTOR_LORA_DROPOUT}" \
    --policy.skill_predictor_reader_tokens="${SKILL_PREDICTOR_READER_TOKENS}" \
    --policy.skill_predictor_reader_depth="${SKILL_PREDICTOR_READER_DEPTH}" \
    --policy.skill_predictor_reader_heads="${SKILL_PREDICTOR_READER_HEADS}" \
    --policy.skill_predictor_deadzone_frac="${SKILL_PREDICTOR_DEADZONE_FRAC}" \
    --policy.skill_predictor_attend_image="${SKILL_PREDICTOR_ATTEND_IMAGE}" \
    --policy.skill_predictor_attend_language="${SKILL_PREDICTOR_ATTEND_LANGUAGE}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.skill_code_space_id="${SKILL_CODE_SPACE_ID}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.proprio_grounding="${PROPRIO_GROUNDING}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.mask_actions_after_skill_end="${MASK_ACTIONS_AFTER_SKILL_END}" \
    --policy.cumulative_xyz_loss_enabled="${CUMULATIVE_XYZ_LOSS_ENABLED}" \
    --policy.cumulative_xyz_loss_weight="${CUMULATIVE_XYZ_LOSS_WEIGHT}" \
    --policy.skill_flow_enabled="${SKILL_FLOW_ENABLED}" \
    --policy.skill_flow_weight="${SKILL_FLOW_WEIGHT}" \
    --policy.skill_flow_max_length="${SKILL_FLOW_MAX_LENGTH}" \
    --policy.skill_flow_target="${SKILL_FLOW_TARGET}" \
    --policy.skill_flow_state_conditioned="${SKILL_FLOW_STATE_CONDITIONED}" \
    --policy.skill_flow_chunk_multiplier="${SKILL_FLOW_CHUNK_MULTIPLIER}" \
    --policy.skill_flow_latent_best_of_n_enabled="${SKILL_FLOW_LATENT_BEST_OF_N_ENABLED}" \
    --policy.skill_flow_latent_candidates="${SKILL_FLOW_LATENT_CANDIDATES}" \
    --policy.skill_flow_latent_top_k="${SKILL_FLOW_LATENT_TOP_K}" \
    --policy.skill_flow_latent_assignment_timesteps="${SKILL_FLOW_LATENT_ASSIGNMENT_TIMESTEPS}" \
    --policy.skill_flow_latent_ranking_route="${SKILL_FLOW_LATENT_RANKING_ROUTE}" \
    --policy.skill_flow_latent_fp32="${SKILL_FLOW_LATENT_FP32}" \
    --policy.skill_flow_latent_dim="${SKILL_FLOW_LATENT_DIM}" \
    --policy.skill_flow_latent_distribution="${SKILL_FLOW_LATENT_DISTRIBUTION}" \
    --policy.skill_flow_latent_gain_init="${SKILL_FLOW_LATENT_GAIN_INIT}" \
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
    --policy.use_muon="${USE_MUON}" \
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
