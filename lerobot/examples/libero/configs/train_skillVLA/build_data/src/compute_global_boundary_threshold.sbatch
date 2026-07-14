#!/usr/bin/env bash
# Reduce the curve-only SkillVLA array into one source-dataset global threshold.
# FT reference mode skips this job and consumes the PT JSON directly.
#SBATCH --job-name=skillVLA_global_mean
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"

EXPECTED_EPISODES=${EXPECTED_EPISODES:?EXPECTED_EPISODES must be supplied}
python "${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/build_data/src/compute_global_boundary_threshold.py" \
  --curves_dir "${SKILLSET_DIR}/curves" \
  --output_path "${SKILLSET_GLOBAL_THRESHOLD_PATH}" \
  --expected_episodes "${EXPECTED_EPISODES}" \
  --dp_run_name "${DP_POLICY_NAME}" \
  --dp_checkpoint "${DP_CHECKPOINT}" \
  --eval_at_step "${SKILLSET_DN_STEP}" \
  --n_gmm_components "${SKILLSET_N_GMM}" \
  --replan_interval "${SKILLSET_REPLAN_INTERVAL}" \
  --smooth_window "${SKILLSET_SMOOTH_WINDOW}" \
  --savgol_polyorder "${SKILLSET_SAVGOL_POLYORDER}"
