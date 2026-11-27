#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:?config}" PHASE="${2:?pt or ft}" COMPONENT="${3:?component}"
REPO_DIR="${PROJECT_ROOT:?resolved project root required}"
PYTHON="${REPO_DIR}/.venv_groot_lerobot/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Run bash setup_env.sh --groot-only first.' >&2; exit 1; }
source "${REPO_DIR}/lerobot/skillVLA/libero/configs/src/runtime_env.sh"
source "${REPO_DIR}/lerobot/skillVLA/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "$PYTHON"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${REPO_DIR}/lerobot/src"
export OMP_NUM_THREADS="${STAGE1_CPU_THREADS:-1}" MKL_NUM_THREADS="${STAGE1_CPU_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_DIR"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS:-1}" \
  "${REPO_DIR}/lerobot/skillVLA/libero/configs/train_skillVLA/PT/src/groot/train.py" \
  --config "$CONFIG_PATH" --phase "$PHASE" --component "$COMPONENT"
