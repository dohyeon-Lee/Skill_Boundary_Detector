#!/usr/bin/env bash
# Step 3 of SkillVLA data generation: segment the source dataset into skills
# using the trained Diffusion Policy (DP) + per-episode DINO features.
# Per-episode DINO is sliced beforehand by submit_build_skillset.sh.
#
# Runs as a Slurm array (one array task per shard of SKILLSET_TASKS_PER_JOB tasks,
# each on its own GPU). The submit script sizes --array from the source task count
# and exports ALL_TASK_IDS; this script picks its own shard from SLURM_ARRAY_TASK_ID.
#
# Output: {skillvla_dataset}/{source}/_work/seg_{dp}_ck{ckpt}/skillset/skills/ep*_skill*.npz
#SBATCH --job-name=skillVLA_skillset
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy, so anchor on SLURM_SUBMIT_DIR
# (the train_skillVLA dir the submit script cd'd into).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""
mkdir -p "${SKILLSET_DIR}"
export WANDB_DIR="${SKILLSET_DIR}"

echo "START SKILLVLA SKILLSET (DP segmentation)"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "Source  : ${RAW_DATASET_DIR}"
echo "DP      : ${DP_POLICY_PATH}"
echo "Output  : ${SKILLSET_DIR}"
echo "Probe   : ${SKILLSET_MODE} (n=${SKILLSET_PROBE_COUNT}, alpha=${SKILLSET_PROBE_ALPHA})"
nvidia-smi || true

if [ ! -d "${DP_POLICY_PATH}" ]; then
  echo "DP policy not found: ${DP_POLICY_PATH}" >&2
  exit 1
fi

# ── pick this array task's task shard ─────────────────────────────────────────
# The submit script exports ALL_TASK_IDS (sorted source task indices) and submits
# with --array=0-(num_shards-1). Each array task takes a contiguous slice of
# SKILLSET_TASKS_PER_JOB ids; --resume makes overlaps/re-runs skip finished work.
# With no array (ALL_TASK_IDS unset) every task runs on a single GPU as before.
TASK_ARGS=()
if [ -n "${ALL_TASK_IDS:-}" ]; then
  IDS=(${ALL_TASK_IDS})
  TPJ="${SKILLSET_TASKS_PER_JOB:-5}"
  AID="${SLURM_ARRAY_TASK_ID:-0}"
  START=$(( AID * TPJ ))
  SHARD_IDS=("${IDS[@]:START:TPJ}")
  if [ "${#SHARD_IDS[@]}" -eq 0 ]; then
    echo "Array task ${AID}: empty shard (start=${START}, total=${#IDS[@]}); nothing to do."
    exit 0
  fi
  echo "Array task ${AID}: tasks ${SHARD_IDS[*]}  (${#SHARD_IDS[@]} of ${#IDS[@]})"
  TASK_ARGS=(--task_ids "${SHARD_IDS[@]}")
fi

# Backfill the per-episode multimodality curves (eval HTML overlay) for an already-built
# run without rewriting skills: submit with CURVES_ONLY=true (resume keyed by curve file).
EXTRA_ARGS=()
[ "${CURVES_ONLY:-false}" = "true" ] && EXTRA_ARGS+=(--curves_only)
[ "${USE_CACHED_CURVES:-false}" = "true" ] && EXTRA_ARGS+=(--use_cached_curves)

# Global threshold selection is resolved from skillvla_data_mode by the config
# emitter. Curves-only jobs deliberately do not read the threshold yet.
EFFECTIVE_GLOBAL_THRESHOLD_PATH="${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-${SKILLSET_GLOBAL_THRESHOLD_PATH}}"
THRESHOLD_ARGS=(
  --boundary_threshold_mode "${SKILLSET_BOUNDARY_THRESHOLD_MODE}"
  --global_threshold_path "${EFFECTIVE_GLOBAL_THRESHOLD_PATH}"
)

DINO_FEATURE_ARGS=()
if [ -n "${SKILLSET_DINO_FEATURE_DIR:-}" ]; then
  DINO_FEATURE_ARGS=(--dino_feature_dir "${SKILLSET_DINO_FEATURE_DIR}")
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/build_skill_dataset.py \
    --dataset_dir          "${RAW_DATASET_DIR}" \
    --policy_path          "${DP_POLICY_PATH}" \
    --output_dir           "${SKILLSET_DIR}" \
    --replan_interval      "${SKILLSET_REPLAN_INTERVAL}" \
    --nms_dist             "${SKILLSET_NMS_DIST}" \
    --min_skills           "${SKILLSET_MIN_SKILLS}" \
    --noise_scheduler_type DDIM \
    --num_inference_steps  10 \
    --eval_at_step         "${SKILLSET_DN_STEP}" \
    --n_gmm_components     "${SKILLSET_N_GMM}" \
    --probe_mode           "${SKILLSET_MODE}" \
    --probe_type           "${SKILLSET_PROBE_TYPE}" \
    --probe_count          "${SKILLSET_PROBE_COUNT}" \
    --probe_alpha          "${SKILLSET_PROBE_ALPHA}" \
    --pca_variance         "${SKILLSET_PCA_VARIANCE}" \
    --pca_stride           "${SKILLSET_PCA_STRIDE}" \
    --pca_scale_mode       "${SKILLSET_PCA_SCALE_MODE}" \
    --probe_exclude_indices "${SKILLSET_PROBE_EXCLUDE_INDICES}" \
    --action_mode          "${SKILLSET_ACTION_MODE}" \
    --relative_exclude_joints "${SKILLSET_RELATIVE_EXCLUDE_JOINTS}" \
    --gripper_mode         "${SKILLSET_GRIPPER_MODE}" \
    --gripper_indices      "${SKILLSET_GRIPPER_INDICES}" \
    --gripper_values       "${SKILLSET_GRIPPER_VALUES}" \
    --gripper_threshold    "${SKILLSET_GRIPPER_THRESHOLD}" \
    --smooth_window        "${SKILLSET_SMOOTH_WINDOW}" \
    --savgol_polyorder     "${SKILLSET_SAVGOL_POLYORDER}" \
    "${THRESHOLD_ARGS[@]}" \
    ${DINO_FEATURE_ARGS[@]+"${DINO_FEATURE_ARGS[@]}"} \
    --resume \
    --write_done_markers \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"}

echo "DONE → ${SKILLSET_DIR}/skills"
