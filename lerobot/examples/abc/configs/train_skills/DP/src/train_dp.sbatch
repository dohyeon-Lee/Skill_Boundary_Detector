#!/usr/bin/env bash
#SBATCH --job-name=ABC_DP
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

if [ -z "${PROJECT_ROOT:-}" ] || [ -z "${DP_DATASET_DIR:-}" ] || [ -z "${DP_OUTPUT_DIR:-}" ]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  CONFIG_PATH="${ABC_DP_CONFIG:-${SUBMIT_DIR}/dp_config.yaml}"
  [ -f "${CONFIG_PATH}" ] || CONFIG_PATH="${SUBMIT_DIR}/../dp_config.yaml"
  [ -f "${CONFIG_PATH}" ] || { echo "dp_config.yaml not found." >&2; exit 1; }
  BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
  [ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
  eval "$("${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/src/train_dp_config.py" \
    --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"

echo "START ABC DP"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${DP_DATASET_DIR}"
echo "tasks    : ${DP_RESOLVED_TASKS_LABEL}"
echo "output   : ${DP_OUTPUT_DIR}"
echo "space    : state/action=14D absolute joints; grounding=off; relative=off"
echo "temporal : obs=${DP_N_OBS_STEPS}, predict=${DP_FUTURE_ACTION_HORIZON} @ 30 Hz"
nvidia-smi || true

[ -d "${DP_DATASET_DIR}" ] || { echo "Dataset not found: ${DP_DATASET_DIR}" >&2; exit 1; }
[ -f "${DP_BASE_CONFIG}" ] || { echo "Base config not found: ${DP_BASE_CONFIG}" >&2; exit 1; }

RESUME_ARGS=()
CHECKPOINT_CONFIG="${DP_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  echo "Checkpoint found. Resuming from ${CHECKPOINT_CONFIG}"
  RESUME_ARGS=(--resume=true --config_path="${CHECKPOINT_CONFIG}")
elif [ -d "${DP_OUTPUT_DIR}" ]; then
  echo "Output exists without a resumable checkpoint; preserving it: ${DP_OUTPUT_DIR}" >&2
  exit 1
else
  RESUME_ARGS=(--config_path="${DP_BASE_CONFIG}")
fi

TASK_ARGS=()
if [ "${DP_TASK_FILTER}" = true ]; then
  TASK_ARGS+=(--dataset.tasks="${DP_RESOLVED_TASKS}")
fi

export ACCELERATE_MIXED_PRECISION=no
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  lerobot-train \
  "${RESUME_ARGS[@]}" \
  --dataset.repo_id="lerobot/${DP_DATASET}" \
  --dataset.root="${DP_DATASET_DIR}" \
  "${TASK_ARGS[@]}" \
  --dataset.use_imagenet_stats=false \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.state_only=true \
  --policy.history_conditioning=state \
  --policy.history_encoder="${DP_HISTORY_ENCODER}" \
  --policy.history_encoder_dim="${DP_HISTORY_ENCODER_DIM}" \
  --policy.n_obs_steps="${DP_N_OBS_STEPS}" \
  --policy.horizon="${DP_TRAJECTORY_HORIZON}" \
  --policy.n_action_steps="${DP_N_ACTION_STEPS}" \
  --policy.action_sequence_mode=future_only \
  --policy.drop_n_last_frames="${DP_DROP_N_LAST_FRAMES}" \
  --policy.down_dims="${DP_DOWN_DIMS_ARG}" \
  --policy.use_amp=false \
  --policy.scheduler_name="${DP_LR_SCHEDULER}" \
  --policy.optimizer_lr="${DP_LR}" \
  --policy.proprio_grounding=none \
  --policy.use_relative_actions=false \
  --output_dir="${DP_OUTPUT_DIR}" \
  --job_name="${DP_RUN_NAME}" \
  --wandb.enable="${DP_WANDB_ENABLE}" \
  --wandb.project="${DP_WANDB_PROJECT}" \
  --steps="${DP_STEPS}" \
  --batch_size="${DP_BATCH_SIZE}" \
  --num_workers="${DP_NUM_WORKERS}" \
  --seed="${DP_SEED}" \
  --log_freq="${DP_LOG_FREQ}" \
  --save_freq="${DP_SAVE_FREQ}" \
  --eval_freq=0

echo "DONE -> ${DP_OUTPUT_DIR}"
