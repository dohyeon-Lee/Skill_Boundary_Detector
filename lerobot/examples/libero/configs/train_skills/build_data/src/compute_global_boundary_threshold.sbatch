#!/usr/bin/env bash
# Compute one global mean from all staged per-episode divergence curves.
# Submitted after the curves-only array and before the cached-curve segmentation array.
#SBATCH --job-name=global_boundary_mean
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Slurm executes a spool copy, so BASH_SOURCE points under /var/spool. Anchor
# paths on the directory from which submit_build_data.sh invoked sbatch instead.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${SUBMIT_DIR}/src"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"

EXPECTED_EPISODES=${EXPECTED_EPISODES:?EXPECTED_EPISODES must be supplied by submit_build_data.sh}
python "${SCRIPT_DIR}/compute_global_boundary_threshold.py" \
  --curves_dir "${SKILLSET_DIR}/curves" \
  --output_path "${SKILLSET_GLOBAL_THRESHOLD_PATH}" \
  --expected_episodes "${EXPECTED_EPISODES}" \
  --dp_run_name "${DP_POLICY}" \
  --dp_checkpoint "${DP_CHECKPOINT}" \
  --eval_at_step "${SKILLSET_DN_STEP}" \
  --n_gmm_components "${SKILLSET_N_GMM}" \
  --replan_interval "${SKILLSET_REPLAN_INTERVAL}" \
  --smooth_window "${SKILLSET_SMOOTH_WINDOW}" \
  --savgol_polyorder "${SKILLSET_SAVGOL_POLYORDER}" \
  --threshold_scale "${SKILLSET_BOUNDARY_THRESHOLD_SCALE}"
