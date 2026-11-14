#!/usr/bin/env bash
#SBATCH --job-name=ABC_SBD_eval
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
EVAL_DIR="${ABC_SBD_EVAL_DIR:?launch with submit.sh}"
CONFIG="${ABC_SBD_EVAL_CONFIG:?missing frozen config}"
BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/config.py" --config "${CONFIG}" --shell)"

cd "${PROJECT_ROOT}/lerobot"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"
export LD_LIBRARY_PATH=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-abc_sbd}_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/src" \
  python -u "${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/src/evaluate.py" \
    --settings-json "${DP_ABLATION_SETTINGS_JSON}"
