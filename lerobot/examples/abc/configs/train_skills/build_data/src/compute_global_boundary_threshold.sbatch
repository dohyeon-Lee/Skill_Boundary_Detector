#!/usr/bin/env bash
#SBATCH --job-name=ABC_global_boundary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_BUILD_DATA_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../src/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"
python examples/libero/configs/train_skills/build_data/src/compute_global_boundary_threshold.py \
  --curves_dir "${SKILLSET_DIR}/curves" \
  --output_path "${SKILLSET_GLOBAL_THRESHOLD_PATH}" \
  --expected_episodes "${ABC_EXPECTED_EPISODES}" \
  --dp_run_name "${DP_POLICY}" --dp_checkpoint "${DP_CHECKPOINT}" \
  --eval_at_step "${SKILLSET_DN_STEP}" --n_gmm_components "${SKILLSET_N_GMM}" \
  --replan_interval "${SKILLSET_REPLAN_INTERVAL}" \
  --smooth_window "${SKILLSET_SMOOTH_WINDOW}" \
  --savgol_polyorder "${SKILLSET_SAVGOL_POLYORDER}" \
  --probe_mode "${SKILLSET_MODE}" --probe_generation "${SKILLSET_PROBE_GENERATION}" \
  --probe_count "${SKILLSET_PROBE_COUNT}" --denoise_output "${SKILLSET_DENOISE_OUTPUT}" \
  --score_metric "${SKILLSET_SCORE_METRIC}" --score_power "${SKILLSET_SCORE_POWER}" \
  --terminal_mask_frames "${SKILLSET_TERMINAL_MASK_FRAMES}" \
  --threshold_scale "${SKILLSET_BOUNDARY_THRESHOLD_SCALE}"
