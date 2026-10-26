#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:?config path required}"
SMOKE="${2:-false}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/../../../../../../.." && pwd)"
BOOTSTRAP_PYTHON="${GROOT_BOOTSTRAP_PYTHON:-${REPO_DIR}/.venv_groot_lerobot/bin/python}"
OPTIONS=()
if [ "${SMOKE}" = true ]; then OPTIONS+=(--smoke); fi
EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/groot_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}" --shell --check)"
eval "${EXPORTS}"
echo "normalization statistics: ${GROOT_NORMALIZATION_STATS}"
export PATH="$(dirname "${GROOT_PYTHON}"):${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${GROOT_PYTHON}"
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"
exec "${GROOT_PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${GROOT_NUM_GPUS}" \
  "${SRC_DIR}/train.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}"
