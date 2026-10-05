#!/usr/bin/env bash
#SBATCH --job-name=newtask_ft_joint
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${NEWTASK_FT_JOINT_CONFIG:-${SRC_DIR}/../joint_ft_config.yaml}"
RESOLVER="${NEWTASK_FT_JOINT_RESOLVER:-${SRC_DIR}/newtask_ft_joint_config.py}"
BOOTSTRAP_PYTHON="${NEWTASK_FT_BOOTSTRAP_PYTHON:-python3}"
eval "$("${BOOTSTRAP_PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START NEWTASK FT (JOINT)"
echo "VSA warm start      : ${VSA_CHECKPOINT_PATH}"
echo "Predictor warm start: ${PREDICTOR_CHECKPOINT_PATH}"
echo "architecture        : ${ARCHITECTURE_LABEL}"
echo "loss                : gt_skill=${JOINT_GT_SKILL_WEIGHT} gt_xyz=${JOINT_GT_XYZ_WEIGHT} hard=${JOINT_ROUTE_HARD_WEIGHT} ste=${JOINT_ROUTE_STE_WEIGHT}"
echo "gradient routes     : action->predictor=${JOINT_ACTION_TO_PREDICTOR} xyz->skill=${JOINT_XYZ_TO_SKILL}"
echo "output              : ${OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing NewTask FT dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/stage_skillvla_dataset.sh"
stage_skillvla_dataset_on_node "${SKILLVLA_DATASET_DIR}" "NewTask Joint FT dataset"
SKILLVLA_DATASET_DIR="${STAGED_SKILLVLA_DATASET_DIR}"

CONFIG_JSON="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CONFIG_JSON}" ]; then
  TRAIN_ARGS=(
    --resume=true
    --config_path="${CONFIG_JSON}"
    --dataset.root="${SKILLVLA_DATASET_DIR}"
    --steps="${STEPS}"
    --log_freq="${LOG_FREQ}"
    --save_freq="${SAVE_FREQ}"
  )
elif [ -e "${OUTPUT_DIR}" ]; then
  echo "Output exists without a resumable checkpoint: ${OUTPUT_DIR}" >&2
  exit 1
else
  TRAIN_ARGS=(
    --dataset.repo_id="${REPO_ID}"
    --dataset.root="${SKILLVLA_DATASET_DIR}"
    --dataset.video_backend=pyav
    --policy.path="${VSA_CHECKPOINT_PATH}"
    --policy.newtask_ft_enabled=true
    --policy.newtask_ft_unfreeze_action_head=false
    --policy.newtask_ft_full_unfreeze=false
    --policy.newtask_ft_skill_flow_loss=false
    --policy.newtask_joint_enabled=true
    --policy.train_skill_predictor=true
    --policy.training_skill_source=gt
    --policy.skill_predictor_checkpoint_path="${PREDICTOR_CHECKPOINT_PATH}"
    --policy.skill_predictor_freeze_vlm="${SKILL_PREDICTOR_FREEZE_VLM}"
    --policy.skill_predictor_detach_vlm="${SKILL_PREDICTOR_DETACH_VLM}"
    --policy.skill_predictor_lora="${SKILL_PREDICTOR_LORA}"
    --policy.skill_predictor_vlm_variant="${SKILL_PREDICTOR_VLM_VARIANT}"
    --policy.skill_predictor_image_size="${SKILL_PREDICTOR_IMAGE_SIZE}"
    --policy.skill_predictor_reader_tokens="${SKILL_PREDICTOR_READER_TOKENS}"
    --policy.skill_predictor_reader_depth="${SKILL_PREDICTOR_READER_DEPTH}"
    --policy.skill_predictor_reader_heads="${SKILL_PREDICTOR_READER_HEADS}"
    --policy.skill_predictor_all_layers="${SKILL_PREDICTOR_ALL_LAYERS}"
    --policy.skill_predictor_deadzone_frac="${SKILL_PREDICTOR_DEADZONE_FRAC}"
    --policy.skill_predictor_attend_image="${SKILL_PREDICTOR_ATTEND_IMAGE}"
    --policy.skill_predictor_attend_language="${SKILL_PREDICTOR_ATTEND_LANGUAGE}"
    --policy.skill_predictor_focus_uv_enabled="${SKILL_PREDICTOR_FOCUS_UV_ENABLED}"
    --policy.skill_predictor_end_state_mode="${SKILL_PREDICTOR_END_STATE_MODE}"
    --policy.skill_predictor_end_state_dim="${SKILL_PREDICTOR_END_STATE_DIM}"
    --policy.skill_predictor_start_proprio=true
    --policy.skill_predictor_start_proprio_dim="${SKILL_PREDICTOR_START_PROPRIO_DIM}"
    --policy.newtask_joint_gt_skill_weight="${JOINT_GT_SKILL_WEIGHT}"
    --policy.newtask_joint_gt_xyz_weight="${JOINT_GT_XYZ_WEIGHT}"
    --policy.newtask_joint_route_hard_weight="${JOINT_ROUTE_HARD_WEIGHT}"
    --policy.newtask_joint_route_ste_weight="${JOINT_ROUTE_STE_WEIGHT}"
    --policy.newtask_joint_route_timesteps="${JOINT_ROUTE_TIMESTEPS}"
    --policy.newtask_joint_action_to_predictor="${JOINT_ACTION_TO_PREDICTOR}"
    --policy.newtask_joint_xyz_to_skill="${JOINT_XYZ_TO_SKILL}"
    --policy.newtask_joint_predictor_lr_scale="${JOINT_PREDICTOR_LR_SCALE}"
    --policy.fsq_path="${FSQ_PATH}"
    --policy.dino_model_path="${DINO_MODEL_PATH}"
    --policy.tokenizer_path="${TOKENIZER_PATH}"
    --policy.tokenizer_max_length="${TOKENIZER_MAX_LENGTH}"
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
    --policy.dino_lr_scale="${DINO_LR_SCALE}"
    --policy.transition_jitter_pmax="${TRANSITION_JITTER_PMAX}"
    --policy.transition_jitter_early_start_pmax="${TRANSITION_JITTER_EARLY_START_PMAX}"
    --policy.transition_jitter_late_start_pmax="${TRANSITION_JITTER_LATE_START_PMAX}"
    --policy.transition_jitter_early_end_pmax="${TRANSITION_JITTER_EARLY_END_PMAX}"
    --policy.transition_jitter_late_end_pmax="${TRANSITION_JITTER_LATE_END_PMAX}"
    --policy.transition_jitter_distribution="${TRANSITION_JITTER_DISTRIBUTION}"
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
    --policy.vsa_debug_steps=0
    --policy.vsa_debug_schedule="${VSA_DEBUG_SCHEDULE}"
    --policy.dtype=bfloat16
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.compile_model=false
    --policy.optimizer_lr="${LR}"
    --policy.scheduler_mode="${SCHEDULER_MODE}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${RUN_NAME}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --steps="${STEPS}"
    --log_freq="${LOG_FREQ}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq=0
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${WANDB_PROJECT}"
  )
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" "${TRAIN_ARGS[@]}"

echo "DONE -> ${OUTPUT_DIR}"
