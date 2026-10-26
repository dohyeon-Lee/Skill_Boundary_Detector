#!/usr/bin/env bash
#SBATCH --job-name=S1xyz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
set -euo pipefail

PROJECT_ROOT="${XYZ_HEAD_PROJECT_ROOT:?Submit with submit_xyz_head_eval.sh}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
if [ "${XYZ_HEAD_DEVICE}" = cuda ]; then
  source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
  require_cuda_or_requeue "${PYTHON}"
fi
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -u "${XYZ_HEAD_EVAL_SRC}/xyz_head_eval.py" \
  --config "${XYZ_HEAD_EVAL_CONFIG}" \
  --worker-index "${SLURM_ARRAY_TASK_ID:-${XYZ_HEAD_WORKER_INDEX:-0}}" \
  --worker-count "${XYZ_HEAD_WORKER_COUNT:-1}"
