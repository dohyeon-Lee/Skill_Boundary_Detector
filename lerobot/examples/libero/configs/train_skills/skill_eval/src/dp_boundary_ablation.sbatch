#!/usr/bin/env bash
#SBATCH --job-name=DPbound
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

EVAL_DIR="${DP_ABLATION_DIR:?launch via submit_dp_boundary_ablation.sh}"
CONFIG="${DP_ABLATION_CONFIG:?missing frozen config}"
BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/dp_boundary_ablation_config.py" --config "${CONFIG}" --shell)"

cd "${PROJECT_ROOT}/lerobot"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export LD_LIBRARY_PATH=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-dp_boundary}_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero" \
  python -u "${EVAL_DIR}/src/dp_boundary_ablation_eval.py" \
    --settings-json "${DP_ABLATION_SETTINGS_JSON}"
