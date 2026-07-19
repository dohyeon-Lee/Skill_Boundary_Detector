#!/usr/bin/env bash
# Step 3b of SkillVLA data generation: after the skillset Slurm array finishes,
# verify every expected episode was segmented and re-run any tasks that still have
# missing episodes (e.g. an array element died on a bad GPU). Submitted with
# --dependency=afterany on the array, so it runs even when some array tasks failed.
#
# Loops up to SKILLSET_MAX_SWEEPS times (each sweep re-runs only the missing tasks,
# on this one GPU, with --resume so finished episodes are skipped). Exits non-zero
# if still incomplete, halting the downstream encode/build chain instead of
# building SkillVLA data on an incomplete skillset.
#SBATCH --job-name=skillVLA_verify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

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

VERIFY="${SRC_DIR}/verify_skillset.py"
MAX_SWEEPS="${SKILLSET_MAX_SWEEPS:-2}"
DINO_FEATURE_ARGS=()
if [ -n "${SKILLSET_DINO_FEATURE_DIR:-}" ]; then
  DINO_FEATURE_ARGS=(--dino_feature_dir "${SKILLSET_DINO_FEATURE_DIR}")
fi

echo "START SKILLVLA SKILLSET VERIFY"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "Skillset: ${SKILLSET_DIR}"
echo "Max sweeps: ${MAX_SWEEPS}"

run_verify () { python -u "${VERIFY}" --dataset_dir "${RAW_DATASET_DIR}" --skillset_dir "${SKILLSET_DIR}"; }
is_complete () { python "${VERIFY}" --dataset_dir "${RAW_DATASET_DIR}" --skillset_dir "${SKILLSET_DIR}" --check; }
missing_tasks () { python "${VERIFY}" --dataset_dir "${RAW_DATASET_DIR}" --skillset_dir "${SKILLSET_DIR}" --print-missing-tasks; }

run_verify || true

sweep=0
while ! is_complete; do
  if [ "${sweep}" -ge "${MAX_SWEEPS}" ]; then
    echo "Skillset still INCOMPLETE after ${MAX_SWEEPS} sweep(s) — failing." >&2
    run_verify >&2 || true
    exit 1
  fi
  sweep=$(( sweep + 1 ))
  MISSING="$(missing_tasks)"
  if [ -z "${MISSING}" ]; then
    echo "No missing tasks reported but --check failed; stopping to avoid a loop." >&2
    exit 1
  fi
  echo "=== sweep ${sweep}/${MAX_SWEEPS}: re-running tasks ${MISSING} ==="
  nvidia-smi || true
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
      --boundary_threshold_mode "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" \
      --global_threshold_path "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-${SKILLSET_GLOBAL_THRESHOLD_PATH}}" \
      ${DINO_FEATURE_ARGS[@]+"${DINO_FEATURE_ARGS[@]}"} \
      --use_cached_curves \
      --resume \
      --write_done_markers \
      --task_ids ${MISSING}
done

echo "=== skillset COMPLETE ==="
run_verify || true
