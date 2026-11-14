#!/usr/bin/env bash
#SBATCH --job-name=ABC_DP_skill
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_BUILD_DATA_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../src/train_skills_config.py" \
  --config "${CONFIG_PATH}" --stage build_data --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"

[ -d "${RAW_DATASET_DIR}" ] || { echo "Dataset not found: ${RAW_DATASET_DIR}" >&2; exit 1; }

# In shared-GPU mode this top-level process owns the one allocation and starts
# bounded child workers.  Each child comes back through this script with an
# explicit logical shard index, so it skips this launcher branch.
if [ "${ABC_SINGLE_GPU_BUILD:-false}" = true ] && [ -z "${ABC_WORKER_SHARD_INDEX:-}" ]; then
  [ "${SKILLSET_SINGLE_GPU_WORKERS}" -gt 0 ] || {
    echo "ABC_SINGLE_GPU_BUILD requires skillset_single_gpu_workers > 0." >&2
    exit 2
  }
  nvidia-smi || true
  exec python -u "${SUBMIT_DIR}/src/run_single_gpu_workers.py"
fi

if [ "${USE_CACHED_CURVES:-false}" != true ]; then
  [ -d "${DP_POLICY_PATH}" ] || { echo "DP checkpoint not found: ${DP_POLICY_PATH}" >&2; exit 1; }
  [ -n "${ABC_WORKER_SHARD_INDEX:-}" ] || nvidia-smi || true
fi

TASK_IDS="$(ABC_TASK_INDICES_JSON="${ABC_TASK_INDICES_JSON}" TASKS_PER_JOB="${SKILLSET_TASKS_PER_JOB}" \
  ARRAY_INDEX="${ABC_WORKER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" "${BOOTSTRAP_PYTHON}" - <<'PY'
import json, os
ids = json.loads(os.environ["ABC_TASK_INDICES_JSON"])
size = int(os.environ["TASKS_PER_JOB"])
start = int(os.environ["ARRAY_INDEX"]) * size
print(" ".join(str(value) for value in ids[start:start + size]))
PY
)"
[ -n "${TASK_IDS}" ] || { echo "No prompt task IDs assigned; exiting."; exit 0; }

mkdir -p "${SKILLSET_DIR}"
CURVES_ARG=()
[ "${CURVES_ONLY:-false}" = true ] && CURVES_ARG=(--curves_only)
CACHED_ARG=()
[ "${USE_CACHED_CURVES:-false}" = true ] && CACHED_ARG=(--use_cached_curves)
[ "${SKILLSET_GAUSSIAN_INCLUDE_MEAN}" = true ] \
  && GAUSSIAN_MEAN_ARG=(--gaussian-include-mean) \
  || GAUSSIAN_MEAN_ARG=(--no-gaussian-include-mean)
[ "${SKILLSET_GMM_WEIGHTED}" = true ] \
  && GMM_WEIGHTED_ARG=(--gmm-weighted) \
  || GMM_WEIGHTED_ARG=(--no-gmm-weighted)

SHARD_INDEX="${ABC_WORKER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
echo "ABC SBD shard=${SHARD_INDEX} prompt_task_ids=${TASK_IDS}"
echo "source=${ABC_SOURCE_TASKS_CSV} action=${SKILLSET_ACTION_MODE} mode=${SKILLSET_MODE} cached=${USE_CACHED_CURVES:-false}"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python examples/libero/configs/train_skills/build_data/src/build_skill_dataset.py \
    --dataset_dir "${RAW_DATASET_DIR}" \
    --policy_path "${DP_POLICY_PATH}" \
    --output_dir "${SKILLSET_DIR}" \
    --source_tasks "${ABC_SOURCE_TASKS_CSV}" \
    --task_ids ${TASK_IDS} \
    "${CURVES_ARG[@]}" \
    "${CACHED_ARG[@]}" \
    --replan_interval "${SKILLSET_REPLAN_INTERVAL}" \
    --nms_dist "${SKILLSET_NMS_DIST}" \
    --min_skill_len "${SKILLSET_MIN_SKILL_LEN}" \
    --min_skills "${SKILLSET_MIN_SKILLS}" \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step "${SKILLSET_DN_STEP}" \
    --n_gmm_components "${SKILLSET_N_GMM}" \
    --probe_mode "${SKILLSET_MODE}" \
    --probe_type "${SKILLSET_PROBE_TYPE}" \
    --probe_count "${SKILLSET_PROBE_COUNT}" \
    --probe_alpha "${SKILLSET_PROBE_ALPHA}" \
    --probe_generation "${SKILLSET_PROBE_GENERATION}" \
    --gaussian_sampling "${SKILLSET_GAUSSIAN_SAMPLING}" \
    "${GAUSSIAN_MEAN_ARG[@]}" \
    --gaussian_seed "${SKILLSET_GAUSSIAN_SEED}" \
    --denoise_steps "${SKILLSET_DENOISE_STEPS}" \
    --denoise_output "${SKILLSET_DENOISE_OUTPUT}" \
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
    --score_metric "${SKILLSET_SCORE_METRIC}" \
    --gmm_covariance "${SKILLSET_GMM_COVARIANCE}" \
    --gmm_n_init "${SKILLSET_GMM_N_INIT}" \
    --gmm_max_iter "${SKILLSET_GMM_MAX_ITER}" \
    --min_effective_samples "${SKILLSET_MIN_EFFECTIVE_SAMPLES}" \
    "${GMM_WEIGHTED_ARG[@]}" \
    --smooth_window "${SKILLSET_SMOOTH_WINDOW}" \
    --savgol_polyorder "${SKILLSET_SAVGOL_POLYORDER}" \
    --score_power "${SKILLSET_SCORE_POWER}" \
    --terminal_mask_frames "${SKILLSET_TERMINAL_MASK_FRAMES}" \
    --boundary_threshold_mode "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" \
    --boundary_threshold_scale "${SKILLSET_BOUNDARY_THRESHOLD_SCALE}" \
    --global_threshold_path "${SKILLSET_GLOBAL_THRESHOLD_PATH}" \
    --write_done_markers \
    --resume \
    --wandb_project "${SKILLSET_WANDB_PROJECT}" \
    --wandb_run_name "abc_${ABC_SOURCE_TASKS_CSV}_shard${SHARD_INDEX}"
