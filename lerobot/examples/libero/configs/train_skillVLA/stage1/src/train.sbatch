#!/usr/bin/env bash
# Stage 1: Arch0/Arch1/Arch2 family training with a trainable pi0.5 action expert.
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
# Slurm runs a spooled copy of this script, so BASH_SOURCE is the spool dir: the submit script
# passes the real src/ (STAGE1_SRC_DIR); a direct `bash train.sbatch` still finds itself.
SRC_DIR="${STAGE1_SRC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SRC_DIR}/../VSA/vsa_train_config.yaml}"
ARCHITECTURE_OVERRIDE="${STAGE1_ARCHITECTURE_OVERRIDE:-}"

BOOTSTRAP_PYTHON="${SRC_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  CONFIG_ARGS+=(--architecture "${ARCHITECTURE_OVERRIDE}")
fi
if ! STAGE1_EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" "${CONFIG_ARGS[@]}")"; then
  echo "Stage-1 config resolution failed inside the job (${SRC_DIR}/stage1_train_config.py)." >&2
  exit 1
fi
eval "${STAGE1_EXPORTS}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE 1"
echo "host    : $(hostname)"
echo "dataset : ${SKILLVLA_DATASET_DIR}"
echo "expert  : pi0.5 base ${PI_BASE} -> full fine-tuning"
echo "arch    : ${ARCHITECTURE_LABEL} (${ARCHITECTURE}, ${ARCHITECTURE_REVISION})"
echo "route   : ${CONDITIONING_ROUTE}"
if [ "${ARCHITECTURE}" = fixed_visual_bottleneck ]; then
  echo "bridge  : fixed ${VISUAL_BOTTLENECK_TOKENS}x${VISUAL_BOTTLENECK_WIDTH} DINO tokens -> final ${VISUAL_BRIDGE_LAST_N_LAYERS} expert layer(s)"
elif [ "${ARCHITECTURE}" = layerwise_cond_bottleneck ]; then
  echo "bridge  : recurrent ${VISUAL_BOTTLENECK_TOKENS}x${VISUAL_BOTTLENECK_WIDTH} Cond-reader tokens -> final ${VISUAL_BRIDGE_LAST_N_LAYERS} expert layer(s)"
fi
echo "base loss: flow (fixed)"
echo "loss mask: after_skill_end=${MASK_ACTIONS_AFTER_SKILL_END}"
echo "proprio : grounding=${PROPRIO_GROUNDING}"
echo "cumulative XYZ: enabled=${CUMULATIVE_XYZ_LOSS_ENABLED} weight=${CUMULATIVE_XYZ_LOSS_WEIGHT}"
echo "mode z  : enabled=${SKILL_FLOW_LATENT_BEST_OF_N_ENABLED} N=${SKILL_FLOW_LATENT_CANDIDATES} K=${SKILL_FLOW_LATENT_TOP_K} M=${SKILL_FLOW_LATENT_ASSIGNMENT_TIMESTEPS} ranking=${SKILL_FLOW_LATENT_RANKING_ROUTE} fp32=${SKILL_FLOW_LATENT_FP32}"
echo "skill   : ${TRAINING_SKILL_SOURCE}   predictor=${SKILL_PREDICTOR_CHECKPOINT_PATH:-none}"
echo "batch   : ordinary random shuffle"
echo "vision  : DINO ${DINO_MODEL_PATH}   frozen=${FREEZE_VISION_ENCODER} lr_scale=${DINO_LR_SCALE}"
echo "foveate : enabled=${FOVEATED_VISION_ENABLED} mode=${FOVEATION_MODE} crop=${FOVEATION_CROP_SIZE}->${FOVEATION_OUTPUT_SIZE}px inner=${FOVEATION_INNER_BOX_MODE}:${FOVEATION_INNER_BOX_SIZE}px"
echo "           shape=${FOVEATION_SHAPE} sharp=${FOVEATION_SHARP_SIZE}px feather=${FOVEATION_FEATHER}px periphery=${FOVEATION_PERIPHERAL_MODE} peripheral_blur=${FOVEATION_PERIPHERAL_BLUR_RADIUS}"
echo "augments: enabled=${FOVEATION_RANDOMIZATION_ENABLED} color=${FOVEATION_COLOR_ENABLED} crop(top-only)=${FOVEATION_CROP_ENABLED} outer=${FOVEATION_CROP_OFFSET_MIN_PX}:${FOVEATION_CROP_OFFSET_MAX_PX}px inner=${FOVEATION_INNER_BOX_OFFSET_MIN_PX}:${FOVEATION_INNER_BOX_OFFSET_MAX_PX}px input_blur(top+wrist)=${FOVEATION_INPUT_BLUR_ENABLED}"
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

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/stage_skillvla_dataset.sh"
stage_skillvla_dataset_on_node "${SKILLVLA_DATASET_DIR}" "Stage-1 dataset"
SKILLVLA_DATASET_DIR="${STAGED_SKILLVLA_DATASET_DIR}"
echo "staged dataset: ${SKILLVLA_DATASET_DIR}"

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
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.cond_encoder_variant="${COND_ENCODER_VARIANT}" \
    --policy.conditioning_route="${CONDITIONING_ROUTE}" \
    --policy.visual_bottleneck_tokens="${VISUAL_BOTTLENECK_TOKENS}" \
    --policy.visual_bottleneck_width="${VISUAL_BOTTLENECK_WIDTH}" \
    --policy.visual_bottleneck_heads="${VISUAL_BOTTLENECK_HEADS}" \
    --policy.visual_bridge_heads="${VISUAL_BRIDGE_HEADS}" \
    --policy.visual_bridge_gate_init="${VISUAL_BRIDGE_GATE_INIT}" \
    --policy.visual_bridge_last_n_layers="${VISUAL_BRIDGE_LAST_N_LAYERS}" \
    --policy.cond_focus_uv_loss_weight="${COND_FOCUS_UV_LOSS_WEIGHT}" \
    --policy.cond_end_xyz_loss_weight="${COND_END_XYZ_LOSS_WEIGHT}" \
    --policy.wrist_patch_align_loss_weight="${WRIST_PATCH_ALIGN_LOSS_WEIGHT}" \
    --policy.bottleneck_termination_loss_weight="${BOTTLENECK_TERMINATION_LOSS_WEIGHT}" \
    --policy.bottleneck_termination_target_sigma="${BOTTLENECK_TERMINATION_TARGET_SIGMA}" \
    --policy.bottleneck_termination_positive_weight="${BOTTLENECK_TERMINATION_POSITIVE_WEIGHT}" \
    --policy.skill_end_pose_mode="${SKILL_END_POSE_MODE}" \
    --policy.vision_backbone=dino \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.dino_image_size="${DINO_IMAGE_SIZE}" \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.dino_lr_scale="${DINO_LR_SCALE}" \
    --policy.foveated_vision_enabled="${FOVEATED_VISION_ENABLED}" \
    --policy.foveation_randomization_enabled="${FOVEATION_RANDOMIZATION_ENABLED}" \
    --policy.foveation_mode="${FOVEATION_MODE}" \
    --policy.foveation_crop_size="${FOVEATION_CROP_SIZE}" \
    --policy.foveation_output_size="${FOVEATION_OUTPUT_SIZE}" \
    --policy.foveation_inner_box_enabled="${FOVEATION_INNER_BOX_ENABLED}" \
    --policy.foveation_inner_box_mode="${FOVEATION_INNER_BOX_MODE}" \
    --policy.foveation_inner_box_size="${FOVEATION_INNER_BOX_SIZE}" \
    --policy.foveation_inner_box_line_width="${FOVEATION_INNER_BOX_LINE_WIDTH}" \
    --policy.foveation_shape="${FOVEATION_SHAPE}" \
    --policy.foveation_sharp_size="${FOVEATION_SHARP_SIZE}" \
    --policy.foveation_feather="${FOVEATION_FEATHER}" \
    --policy.foveation_peripheral_mode="${FOVEATION_PERIPHERAL_MODE}" \
    --policy.foveation_peripheral_blur_radius="${FOVEATION_PERIPHERAL_BLUR_RADIUS}" \
    --policy.foveation_color_enabled="${FOVEATION_COLOR_ENABLED}" \
    --policy.foveation_brightness_min="${FOVEATION_BRIGHTNESS_MIN}" \
    --policy.foveation_brightness_max="${FOVEATION_BRIGHTNESS_MAX}" \
    --policy.foveation_contrast_min="${FOVEATION_CONTRAST_MIN}" \
    --policy.foveation_contrast_max="${FOVEATION_CONTRAST_MAX}" \
    --policy.foveation_saturation_min="${FOVEATION_SATURATION_MIN}" \
    --policy.foveation_saturation_max="${FOVEATION_SATURATION_MAX}" \
    --policy.foveation_hue_min="${FOVEATION_HUE_MIN}" \
    --policy.foveation_hue_max="${FOVEATION_HUE_MAX}" \
    --policy.foveation_crop_enabled="${FOVEATION_CROP_ENABLED}" \
    --policy.foveation_crop_offset_min_px="${FOVEATION_CROP_OFFSET_MIN_PX}" \
    --policy.foveation_crop_offset_max_px="${FOVEATION_CROP_OFFSET_MAX_PX}" \
    --policy.foveation_inner_box_offset_min_px="${FOVEATION_INNER_BOX_OFFSET_MIN_PX}" \
    --policy.foveation_inner_box_offset_max_px="${FOVEATION_INNER_BOX_OFFSET_MAX_PX}" \
    --policy.foveation_input_blur_enabled="${FOVEATION_INPUT_BLUR_ENABLED}" \
    --policy.foveation_input_blur_min_radius="${FOVEATION_INPUT_BLUR_MIN_RADIUS}" \
    --policy.foveation_input_blur_max_radius="${FOVEATION_INPUT_BLUR_MAX_RADIUS}" \
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
