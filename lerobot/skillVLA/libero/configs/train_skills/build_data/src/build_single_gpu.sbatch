#!/usr/bin/env bash
#SBATCH --job-name=LIBERO_DP_skill
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SUBMIT_DIR="${LIBERO_BUILD_DIR:?Missing build directory}"
source "${LIBERO_BUILD_ENV:?Missing frozen build settings}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src/runtime_env.sh"
export SLURM_SUBMIT_DIR="${SUBMIT_DIR}"
# A worker failure must never requeue/cancel the enclosing shared allocation.
export SBD_SRUN_SESSION=1
if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" != 1 ]; then
  python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "SBD build requires one visible GPU"'
fi
exec python -u "${SUBMIT_DIR}/src/run_single_gpu_workers.py"
