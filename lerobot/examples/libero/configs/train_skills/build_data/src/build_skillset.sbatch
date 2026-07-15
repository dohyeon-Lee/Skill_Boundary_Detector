#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
# Reference models:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Outputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}_{mode}/skillset
#
# Build skillset data by segmenting demonstrations with a trained DINO-DP.
#SBATCH --job-name=DP_skill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../build_data_config.yaml}

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

echo "START SKILLSET JOB"
echo "Node          : ${SLURMD_NODENAME:-local}"
echo "Array Task ID : ${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi

if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -d "${DP_POLICY_PATH}" ]; then
  echo "DP policy checkpoint not found: ${DP_POLICY_PATH}" >&2
  exit 1
fi

TASKS_PER_JOB=${SKILLSET_TASKS_PER_JOB}
TOTAL_TASKS=${TOTAL_TASKS:-}
if [ -z "${TOTAL_TASKS}" ]; then
  TOTAL_TASKS=$("${BOOTSTRAP_PYTHON}" - <<PY
from pathlib import Path
import pandas as pd
tasks = pd.read_parquet(Path("${RAW_DATASET_DIR}") / "meta" / "tasks.parquet")
print(int(tasks["task_index"].nunique()))
PY
)
fi

MAX_TASK_ID=$(( TOTAL_TASKS - 1 ))
START_TASK=$(( ${SLURM_ARRAY_TASK_ID:-0} * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))

TASK_IDS=""
for TASK_ID in $(seq "${START_TASK}" "${END_TASK}"); do
  if [ "${TASK_ID}" -gt "${MAX_TASK_ID}" ]; then
    break
  fi
  TASK_IDS="${TASK_IDS} ${TASK_ID}"
done

if [ -z "${TASK_IDS// }" ]; then
  echo "No tasks assigned to this array index; exiting."
  exit 0
fi

mkdir -p "${SKILLSET_DIR}"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-${SKILLSET_DIR}}

WANDB_RUN_NAME=${TARGET_DATASET}_skillset_array${SLURM_ARRAY_TASK_ID:-0}_tasks${START_TASK}-${END_TASK}

echo "Build skillset"
echo "  dataset    : ${TARGET_DATASET}"
echo "  raw data   : ${RAW_DATASET_DIR}"
echo "  policy     : ${DP_POLICY_PATH}"
echo "  output     : ${SKILLSET_DIR}"
echo "  probe mode : ${SKILLSET_MODE} (n=${SKILLSET_PROBE_COUNT}, alpha=${SKILLSET_PROBE_ALPHA})"
echo "  task ids   :${TASK_IDS}"

# Curves-only backfill (CURVES_ONLY=true from submit_build_data.sh): regenerate per-episode
# multimodality curves for an EXISTING skillset; build_skill_dataset leaves skills untouched and only
# dumps curves/ep*.npz (resume keyed on the curve file).
CURVES_ARG=()
if [ "${CURVES_ONLY:-false}" = "true" ]; then
  CURVES_ARG=(--curves_only)
  echo "  mode       : CURVES_ONLY (skills unchanged, dumping curves only)"
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python examples/libero/build_skill_dataset.py \
    --dataset_dir "${RAW_DATASET_DIR}" \
    --policy_path "${DP_POLICY_PATH}" \
    --output_dir "${SKILLSET_DIR}" \
    --task_ids ${TASK_IDS} \
    ${CURVES_ARG[@]+"${CURVES_ARG[@]}"} \
    --replan_interval "${SKILLSET_REPLAN_INTERVAL}" \
    --nms_dist "${SKILLSET_NMS_DIST}" \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step "${SKILLSET_DN_STEP}" \
    --n_gmm_components "${SKILLSET_N_GMM}" \
    --probe_mode "${SKILLSET_MODE}" \
    --probe_type "${SKILLSET_PROBE_TYPE}" \
    --probe_count "${SKILLSET_PROBE_COUNT}" \
    --probe_alpha "${SKILLSET_PROBE_ALPHA}" \
    --pca_variance "${SKILLSET_PCA_VARIANCE}" \
    --pca_stride "${SKILLSET_PCA_STRIDE}" \
    --pca_scale_mode "${SKILLSET_PCA_SCALE_MODE}" \
    --probe_exclude_indices "${SKILLSET_PROBE_EXCLUDE_INDICES}" \
    --action_mode "${SKILLSET_ACTION_MODE}" \
    --relative_exclude_joints "${SKILLSET_RELATIVE_EXCLUDE_JOINTS}" \
    --gripper_mode "${SKILLSET_GRIPPER_MODE}" \
    --gripper_indices "${SKILLSET_GRIPPER_INDICES}" \
    --gripper_values "${SKILLSET_GRIPPER_VALUES}" \
    --gripper_threshold "${SKILLSET_GRIPPER_THRESHOLD}" \
    --smooth_window "${SKILLSET_SMOOTH_WINDOW}" \
    --savgol_polyorder "${SKILLSET_SAVGOL_POLYORDER}" \
    --resume \
    --wandb_project "${SKILLSET_WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}"

echo "DONE array ${SLURM_ARRAY_TASK_ID:-0}"
