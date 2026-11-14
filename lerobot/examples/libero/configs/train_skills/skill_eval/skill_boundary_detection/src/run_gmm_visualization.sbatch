#!/usr/bin/env bash
#SBATCH --job-name=SBD_GMM_viz
#SBATCH --gres=gpu:1
set -euo pipefail

PROJECT_ROOT="${SBD_GMM_PROJECT_ROOT:?missing project root}"
SETTINGS_B64="${SBD_GMM_SETTINGS_B64:?missing settings}"
RUNTIME_ENV="${SBD_GMM_RUNTIME_ENV:-${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh}"

cd "${PROJECT_ROOT}/lerobot"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${RUNTIME_ENV}"
export LD_LIBRARY_PATH=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-sbd_gmm}_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

SETTINGS_JSON="$(printf '%s' "${SETTINGS_B64}" | base64 --decode)"
PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/src" \
  python -u examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/src/evaluate.py \
    --settings-json "${SETTINGS_JSON}"
