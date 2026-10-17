#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:?config path required}"
SMOKE="${2:-false}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/../../../../../../.." && pwd)"
BOOTSTRAP_PYTHON="${MOLMOACT2_BOOTSTRAP_PYTHON:-${REPO_DIR}/.venv_molmoact2/bin/python}"
OPTIONS=()
if [ "${SMOKE}" = true ]; then OPTIONS+=(--smoke); fi
EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/molmoact2_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}" --shell --check)"
eval "${EXPORTS}"
export PATH="$(dirname "${MOLMOACT2_PYTHON}"):${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${MOLMOACT2_PYTHON}"
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"
exec "${MOLMOACT2_PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${MOLMOACT2_NUM_GPUS}" \
  "${SRC_DIR}/train.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}"
