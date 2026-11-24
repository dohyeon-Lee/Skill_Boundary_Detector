#!/usr/bin/env bash
set -euo pipefail
stage="$1"; config="$2"
: "${PROJECT_ROOT:?Submit through the DROID submit_train script}"
export SBD_GLOBAL_CONFIG="$(dirname "${config}")/global_config.yaml"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SBD_GPU_GUARD_TERMINAL_ON_OOM=1
source "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src/runtime_env.sh"
cd "${PROJECT_ROOT}"
exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/lerobot/skillVLA/droid/configs/train_skills/src/train_skills_config.py" --stage "${stage}" --config "${config}" --run
