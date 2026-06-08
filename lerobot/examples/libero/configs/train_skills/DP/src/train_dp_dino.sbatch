#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
# Reference models/configs:
#   base DP cfg  : {project_root}/lerobot/{dp_base_config}
# Outputs:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}
#   checkpoint   : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
#
# Train Diffusion Policy with precomputed frame-level DINO tokens.
#
# Submit through ../submit_train_dp_dino.sh so yaml Slurm settings and DINO
# preparation run first.
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
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../../train_skills_config.yaml}

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
if [ ! -d "${DINO_FEATURE_DIR}" ]; then
  echo "DINO feature dir not found: ${DINO_FEATURE_DIR}" >&2
  echo "Run prepare_dino_for_training_dataset.py for ${TARGET_DATASET} first." >&2
  echo "Base DINO source expected at: ${BASE_DINO_FEATURE_DIR}" >&2
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

DINO_IMAGE_KEYS_JSON=$(python - <<PY
import json
print(json.dumps("${DINO_IMAGE_KEYS}".split(",")))
PY
)
WANDB_ENABLE=${DP_WANDB_ENABLE}

echo "DP DINO train"
echo "  dataset      : ${TARGET_DATASET}"
echo "  raw data     : ${RAW_DATASET_DIR}"
echo "  dino dataset : ${DINO_FEATURE_DATASET}"
echo "  dino dir     : ${DINO_FEATURE_DIR}"
echo "  image keys   : ${DINO_IMAGE_KEYS_JSON}"
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
  --policy.n_obs_steps="${DP_N_OBS_STEPS}" \
  --policy.n_action_steps="${DP_N_ACTION_STEPS}" \
  --policy.horizon="${DP_HORIZON}" \
  --policy.use_dino_features=true \
  --policy.dino_feature_dir="${DINO_FEATURE_DIR}" \
  --policy.dino_image_keys="${DINO_IMAGE_KEYS_JSON}" \
  --policy.dino_patch_grid="${DINO_PATCH_GRID}" \
  --policy.dino_feature_dim="${DINO_FEATURE_DIM}" \
  --policy.dino_visual_feature_dim="${DINO_VISUAL_FEATURE_DIM}" \
  --policy.dino_transformer_n_layers="${DINO_TRANSFORMER_N_LAYERS}" \
  --policy.dino_transformer_n_heads="${DINO_TRANSFORMER_N_HEADS}" \
  --policy.dino_cache_size="${DINO_CACHE_SIZE}" \
  --dataset.repo_id="lerobot/${TARGET_DATASET}" \
  --dataset.root="${RAW_DATASET_DIR}" \
  --dataset.video_backend=pyav \
  --batch_size="${DP_BATCH_SIZE}" \
  --num_workers="${DP_NUM_WORKERS}" \
  ${RESUME_FLAG}

echo "DONE"
