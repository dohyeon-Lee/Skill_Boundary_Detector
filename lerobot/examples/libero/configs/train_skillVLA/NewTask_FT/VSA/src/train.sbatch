#!/usr/bin/env bash
# NewTask FT: adapt a Stage-1 VSA (Arch4--Arch13) checkpoint to a new-task dataset.
#SBATCH --job-name=newtask_ft_vsa
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${NEWTASK_FT_VSA_CONFIG:-${SRC_DIR}/../vsa_ft_config.yaml}"
# Slurm spools this script, so SRC_DIR may be the spool directory.
RESOLVER="${NEWTASK_FT_VSA_RESOLVER:-${SRC_DIR}/newtask_ft_vsa_config.py}"

BOOTSTRAP_PYTHON="${NEWTASK_FT_BOOTSTRAP_PYTHON:-python3}"
eval "$("${BOOTSTRAP_PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START NEWTASK FT (VSA)"
echo "host      : $(hostname)"
echo "warm start: ${VSA_CHECKPOINT_PATH}"
echo "arch      : ${ARCHITECTURE_LABEL} (contract inherited from checkpoint)"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
if [ "${FULL_UNFREEZE}" = true ]; then
  echo "frozen    : nothing in the action model (full fine-tuning baseline)"
  echo "trainable : Cond, bottleneck, bridge, ALL Expert layers, action I/O, time MLP, skill broadcast, auxiliary heads"
  echo "loss      : main action flow + architecture auxiliaries$([ "${SKILL_FLOW_ACTIVE}" = true ] && echo " + skill-flow" || echo " (skill-flow loss off)")"
elif [ "${UNFREEZE_ACTION_HEAD}" = true ]; then
  echo "frozen    : Expert motion core, action input, time MLP, skill broadcast"
  echo "trainable : Cond, bottleneck, bridge, last ${TRAINABLE_EXPERT_LAYERS} Expert layer(s), auxiliary heads, final Expert norm, action head"
  echo "loss      : main action flow + architecture auxiliaries$([ "${SKILL_FLOW_ACTIVE}" = true ] && echo " + skill-flow (trains final norm and action head)" || echo " (skill-flow loss off)")"
else
  echo "frozen    : Expert motion core, final Expert norm, action I/O, time MLP, skill broadcast"
  echo "trainable : Cond, bottleneck, bridge, last ${TRAINABLE_EXPERT_LAYERS} Expert layer(s), auxiliary heads"
  echo "loss      : main action flow + architecture auxiliaries (skill-flow loss skipped)"
fi
echo "dino      : frozen=${FREEZE_VISION_ENCODER} lr_scale=${DINO_LR_SCALE}"
echo "normalize : inherited from the warm-start checkpoint"
echo "schedule  : steps=${STEPS} mode=${SCHEDULER_MODE} warmup=${SCHEDULER_WARMUP_STEPS} lr=${LR}"
echo "output    : ${OUTPUT_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing NewTask FT dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/stage_skillvla_dataset.sh"
stage_skillvla_dataset_on_node "${SKILLVLA_DATASET_DIR}" "NewTask FT dataset"
SKILLVLA_DATASET_DIR="${STAGED_SKILLVLA_DATASET_DIR}"
echo "staged dataset: ${SKILLVLA_DATASET_DIR}"

CONFIG_JSON="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found; resuming from ${CONFIG_JSON}"
  # Model, optimizer, scheduler position, and W&B run all come from the checkpoint. Only what may
  # legitimately change between submissions is re-supplied: the node-local dataset path, and the
  # run length / cadence, so raising schedule.steps in the YAML continues a finished run.
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
    --policy.newtask_ft_unfreeze_action_head="${UNFREEZE_ACTION_HEAD}"
    --policy.newtask_ft_full_unfreeze="${FULL_UNFREEZE}"
    --policy.newtask_ft_skill_flow_loss="${SKILL_FLOW_LOSS}"
    --policy.fsq_path="${FSQ_PATH}"
    --policy.dino_model_path="${DINO_MODEL_PATH}"
    --policy.tokenizer_path="${TOKENIZER_PATH}"
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
