#!/usr/bin/env bash
#SBATCH --job-name=FTpred
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
set -euo pipefail
PROJECT_ROOT="${FT_PRED_EVAL_PROJECT_ROOT:?Submit with submit_pred_eval.sh}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
if [ "${FT_PRED_DEVICE}" = cuda ]; then
  source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
  require_cuda_or_requeue "${PYTHON}"
fi
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -u "${FT_PRED_EVAL_SRC}/ft_pred_eval.py" --config "${FT_PRED_EVAL_CONFIG}" \
  --worker-index "${SLURM_ARRAY_TASK_ID:-${FT_PRED_WORKER_INDEX:-0}}" \
  --worker-count "${FT_PRED_WORKER_COUNT:-1}"
