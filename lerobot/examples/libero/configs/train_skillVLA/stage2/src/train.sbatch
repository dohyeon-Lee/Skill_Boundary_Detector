#!/usr/bin/env bash
# Stage 2: frozen Stage-1 VSA prior + four language-conditioned likelihood blocks.
#SBATCH --job-name=stage2_likelihood
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

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE 2 LIKELIHOOD"
echo "dataset : ${SKILLVLA_DATASET_DIR}"
echo "prior   : ${STAGE1_CHECKPOINT_PATH} (fully frozen)"
echo "train   : VLM projection + ${LIKELIHOOD_NUM_LAYERS} blocks + action head"
echo "skill   : ${TRAINING_SKILL_SOURCE}"
echo "aux     : predictor=${FINETUNE_SKILL_PREDICTOR}, terminator=${FINETUNE_TERMINATOR}"
echo "output  : ${PT_OUTPUT_DIR}"

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

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla_stage2 \
    --policy.stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
    --policy.training_skill_source="${TRAINING_SKILL_SOURCE}" \
    --policy.likelihood_num_layers="${LIKELIHOOD_NUM_LAYERS}" \
    --policy.likelihood_cross_attention_heads="${LIKELIHOOD_CROSS_ATTENTION_HEADS}" \
    --policy.finetune_skill_predictor="${FINETUNE_SKILL_PREDICTOR}" \
    --policy.finetune_terminator="${FINETUNE_TERMINATOR}" \
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
