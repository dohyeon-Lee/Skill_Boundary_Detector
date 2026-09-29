#!/usr/bin/env bash
#SBATCH --job-name=stage1-align-map
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SRC_DIR="${STAGE1_ATTENTION_EVAL_SRC_DIR:?missing STAGE1_ATTENTION_EVAL_SRC_DIR}"
CONFIG_PATH="${ATTENTION_EVAL_CONFIG:?missing ATTENTION_EVAL_CONFIG}"
cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

python "${SRC_DIR}/visualize_attention.py" --config "${CONFIG_PATH}"
