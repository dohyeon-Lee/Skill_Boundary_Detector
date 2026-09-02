#!/usr/bin/env bash
# Stage 2: corrected likelihood refinement or FRS/DSBC initial-noise selection.
#SBATCH --job-name=stage2_likelihood
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
CONFIG_PATH="${STAGE2_TRAIN_CONFIG:-${SUBMIT_DIR}/stage2_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if [ -n "${STAGE2_ENV_SNAPSHOT:-}" ] && [ -f "${STAGE2_ENV_SNAPSHOT}" ]; then
  source "${STAGE2_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_train_config.py" --config "${CONFIG_PATH}" --shell)"
fi
# Environment snapshots created before DSBC existed remain valid likelihood
# snapshots, including across a manual restart after this code update.
STAGE2_MODE="${STAGE2_MODE:-likelihood}"
DSBC_NOISE_OUTPUT_MODE="${DSBC_NOISE_OUTPUT_MODE:-shared}"
DSBC_FRS_NUM_STEPS="${DSBC_FRS_NUM_STEPS:-${NUM_INFERENCE_STEPS:-10}}"
DSBC_ANCHOR_SEED="${DSBC_ANCHOR_SEED:-0}"
PROPRIO_GROUNDING="${PROPRIO_GROUNDING:-none}"
SKILL_FLOW_ENABLED="${SKILL_FLOW_ENABLED:-false}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE 2 (${STAGE2_MODE})"
echo "dataset  : ${SKILLVLA_DATASET_DIR}"
echo "prior    : ${STAGE1_CHECKPOINT_PATH} (fully frozen, ${ARCHITECTURE}/${ARCHITECTURE_REVISION})"
echo "proprio  : grounding=${PROPRIO_GROUNDING}, normalization=Stage-1 checkpoint"
echo "base VLM : ${VLM_BASE_PATH} (fully frozen)"
if [ -n "${PREDICTOR_CHECKPOINT_PATH}" ]; then
  echo "predictor: ${PREDICTOR_CHECKPOINT_PATH} (learned overlay, fully frozen)"
else
  echo "predictor: none (GT skill source)"
fi
if [ "${STAGE2_MODE}" = dsbc ]; then
  echo "train    : VLM projection + ${LIKELIHOOD_NUM_LAYERS} blocks + noise head (Stage-1 head frozen)"
  echo "DSBC     : output=${DSBC_NOISE_OUTPUT_MODE}, online FRS steps=${DSBC_FRS_NUM_STEPS}, anchor seed=${DSBC_ANCHOR_SEED}"
else
  echo "train    : VLM projection + ${LIKELIHOOD_NUM_LAYERS} blocks + action head"
fi
echo "memory   : ${LIKELIHOOD_VLM_MEMORY} (gate lr x${LIKELIHOOD_GATE_LR_SCALE})"
echo "schedule : ${SCHEDULER_MODE} warmup=${SCHEDULER_WARMUP_STEPS}"
echo "skill    : ${TRAINING_SKILL_SOURCE}"
echo "cumulative XYZ: enabled=${CUMULATIVE_XYZ_LOSS_ENABLED} weight=${CUMULATIVE_XYZ_LOSS_WEIGHT}"
echo "batching : batch=${BATCH_SIZE}, same-skill=${SAME_SKILL_BATCH_ENABLED}, grouped=${SAME_SKILL_BATCH_FRACTION}, progress-temp=${SAME_SKILL_PROGRESS_TEMPERATURE}"
echo "output   : ${PT_OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing Stage-2 dataset: ${SKILLVLA_DATASET_DIR}" >&2
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

PREDICTOR_ARGS=()
if [ -n "${PREDICTOR_CHECKPOINT_PATH}" ]; then
  PREDICTOR_ARGS+=(--policy.skill_predictor_checkpoint_path="${PREDICTOR_CHECKPOINT_PATH}")
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla_stage2 \
    --policy.stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
    --policy.vlm_base_path="${VLM_BASE_PATH}" \
    "${PREDICTOR_ARGS[@]}" \
    --policy.architecture="${ARCHITECTURE}" \
    --policy.architecture_revision="${ARCHITECTURE_REVISION}" \
    --policy.architecture_label="${ARCHITECTURE_LABEL}" \
    --policy.proprio_grounding="${PROPRIO_GROUNDING}" \
    --policy.skill_flow_enabled="${SKILL_FLOW_ENABLED}" \
    --policy.stage2_mode="${STAGE2_MODE}" \
    --policy.training_skill_source="${TRAINING_SKILL_SOURCE}" \
    --policy.cumulative_xyz_loss_enabled="${CUMULATIVE_XYZ_LOSS_ENABLED}" \
    --policy.cumulative_xyz_loss_weight="${CUMULATIVE_XYZ_LOSS_WEIGHT}" \
    --policy.same_skill_batch_enabled="${SAME_SKILL_BATCH_ENABLED}" \
    --policy.same_skill_batch_fraction="${SAME_SKILL_BATCH_FRACTION}" \
    --policy.same_skill_progress_temperature="${SAME_SKILL_PROGRESS_TEMPERATURE}" \
    --policy.likelihood_num_layers="${LIKELIHOOD_NUM_LAYERS}" \
    --policy.likelihood_cross_attention_heads="${LIKELIHOOD_CROSS_ATTENTION_HEADS}" \
    --policy.likelihood_vlm_memory="${LIKELIHOOD_VLM_MEMORY}" \
    --policy.likelihood_gate_lr_scale="${LIKELIHOOD_GATE_LR_SCALE}" \
    --policy.dsbc_noise_output_mode="${DSBC_NOISE_OUTPUT_MODE}" \
    --policy.dsbc_frs_num_steps="${DSBC_FRS_NUM_STEPS}" \
    --policy.dsbc_anchor_seed="${DSBC_ANCHOR_SEED}" \
    --policy.scheduler_mode="${SCHEDULER_MODE}" \
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}" \
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.cond_encoder_variant="${COND_ENCODER_VARIANT}" \
    --policy.vision_backbone=dino \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.dino_image_size="${DINO_IMAGE_SIZE}" \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.conditioning_route="${CONDITIONING_ROUTE}" \
    --policy.mask_actions_after_skill_end="${MASK_ACTIONS_AFTER_SKILL_END}" \
    --policy.num_visual_latents_per_camera="${NUM_VISUAL_LATENTS_PER_CAMERA}" \
    --policy.visual_perceiver_width="${VISUAL_PERCEIVER_WIDTH}" \
    --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.transition_jitter_pmax="${TRANSITION_JITTER_PMAX}" \
    --policy.transition_jitter_early_start_pmax="${TRANSITION_JITTER_EARLY_START_PMAX}" \
    --policy.transition_jitter_late_start_pmax="${TRANSITION_JITTER_LATE_START_PMAX}" \
    --policy.transition_jitter_early_end_pmax="${TRANSITION_JITTER_EARLY_END_PMAX}" \
    --policy.transition_jitter_late_end_pmax="${TRANSITION_JITTER_LATE_END_PMAX}" \
    --policy.transition_jitter_distribution="${TRANSITION_JITTER_DISTRIBUTION}" \
    --policy.train_skill_predictor="${TRAIN_SKILL_PREDICTOR}" \
    --policy.skill_predictor_vlm_variant="${SKILL_PREDICTOR_VLM_VARIANT}" \
    --policy.skill_predictor_image_size="${SKILL_PREDICTOR_IMAGE_SIZE}" \
    --policy.skill_predictor_reader_tokens="${SKILL_PREDICTOR_READER_TOKENS}" \
    --policy.skill_predictor_reader_depth="${SKILL_PREDICTOR_READER_DEPTH}" \
    --policy.skill_predictor_reader_heads="${SKILL_PREDICTOR_READER_HEADS}" \
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.skill_predictor_detach_vlm="${SKILL_PREDICTOR_DETACH_VLM}" \
    --policy.skill_predictor_lora="${SKILL_PREDICTOR_LORA}" \
    --policy.skill_predictor_lora_targets="${SKILL_PREDICTOR_LORA_TARGETS}" \
    --policy.skill_predictor_lora_rank="${SKILL_PREDICTOR_LORA_RANK}" \
    --policy.skill_predictor_lora_alpha="${SKILL_PREDICTOR_LORA_ALPHA}" \
    --policy.skill_predictor_lora_dropout="${SKILL_PREDICTOR_LORA_DROPOUT}" \
    --policy.skill_predictor_deadzone_frac="${SKILL_PREDICTOR_DEADZONE_FRAC}" \
    --policy.skill_predictor_attend_image="${SKILL_PREDICTOR_ATTEND_IMAGE}" \
    --policy.skill_predictor_attend_language="${SKILL_PREDICTOR_ATTEND_LANGUAGE}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.skill_code_space_id="${SKILL_CODE_SPACE_ID}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}" \
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
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
