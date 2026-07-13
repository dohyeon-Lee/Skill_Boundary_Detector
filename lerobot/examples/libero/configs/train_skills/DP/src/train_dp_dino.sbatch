#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
# Reference models/configs:
#   base DP cfg  : {project_root}/lerobot/{dp_base_config}
# Outputs:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}
#   checkpoint   : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
#
# Train Diffusion Policy (resnet: raw frames | state: proprioceptive-only).
#
# Submit through ../submit_train_dp_dino.sh so yaml Slurm settings resolve first.
#
# Override dataset/config when needed:
#   TRAIN_DATA=libero_90_full_10 sbatch train_dp_dino.sbatch
#   TRAIN_SKILLS_CONFIG=/path/to/train_skills_config.yaml sbatch train_dp_dino.sbatch
#
#SBATCH --job-name=DP_DINO_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../dp_config.yaml}

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${PROJECT_ROOT}/../.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${PROJECT_ROOT}/../.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START DP DINO TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
nvidia-smi

if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -f "${DP_BASE_CONFIG}" ]; then
  echo "DP base config not found: ${DP_BASE_CONFIG}" >&2
  exit 1
fi

RESUME_FLAG=""
CHECKPOINT_CONFIG="${DP_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${DP_OUTPUT_DIR}" ]; then
  if [ "${DP_OVERWRITE_OUTPUT}" = "true" ]; then
    rm -rf "${DP_OUTPUT_DIR}"
    TRAIN_CONFIG_PATH="${DP_BASE_CONFIG}"
  else
    echo "Output dir already exists and no last checkpoint was found:" >&2
    echo "  ${DP_OUTPUT_DIR}" >&2
    echo "Set dp_overwrite_output: true in ${CONFIG_PATH} if you want to replace it." >&2
    exit 1
  fi
else
  TRAIN_CONFIG_PATH="${DP_BASE_CONFIG}"
fi

WANDB_ENABLE=${DP_WANDB_ENABLE}

# Vision encoder branch: "resnet" = original lerobot Diffusion Policy (ResNet on raw frames;
# image_features come from the dataset's camera keys). "state" = proprioceptive-only (no vision).
POLICY_ARGS=(
  --policy.n_obs_steps="${DP_N_OBS_STEPS}"
  --policy.n_action_steps="${DP_N_ACTION_STEPS}"
  --policy.horizon="${DP_HORIZON}"
)
if [ "${DP_VISION}" = "resnet" ]; then
  echo "DP train (vision=resnet, ORIGINAL Diffusion Policy — no DINO)"
  echo "  backbone     : ${DP_VISION_BACKBONE}"
  POLICY_ARGS+=(
    --policy.vision_backbone="${DP_VISION_BACKBONE}"
  )
elif [ "${DP_VISION}" = "state" ]; then
  echo "DP train (vision=state, proprioceptive-only — no image/DINO input)"
  POLICY_ARGS+=(
    --policy.state_only=true
  )
else
  echo "dp_vision must be 'resnet' or 'state' (got '${DP_VISION}') — DINO encoder 옵션은 은퇴함." >&2
  exit 1
fi
echo "  dataset      : ${TARGET_DATASET}"
echo "  output       : ${DP_OUTPUT_DIR}"
echo "  config       : ${TRAIN_CONFIG_PATH}"
echo "  policy       : ${DP_POLICY}"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  lerobot-train \
  --config_path="${TRAIN_CONFIG_PATH}" \
  --output_dir="${DP_OUTPUT_DIR}" \
  --job_name="${DP_POLICY}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${DP_WANDB_PROJECT}" \
  --steps="${DP_STEPS}" \
  --log_freq="${DP_LOG_FREQ}" \
  --save_freq="${DP_SAVE_FREQ}" \
  --eval_freq="${DP_EVAL_FREQ}" \
  --seed="${DP_SEED}" \
  "${POLICY_ARGS[@]}" \
  --dataset.repo_id="lerobot/${TARGET_DATASET}" \
  --dataset.root="${RAW_DATASET_DIR}" \
  --dataset.video_backend=pyav \
  --batch_size="${DP_BATCH_SIZE}" \
  --num_workers="${DP_NUM_WORKERS}" \
  ${RESUME_FLAG}

echo "DONE"
