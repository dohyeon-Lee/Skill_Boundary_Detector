#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
EVAL_DIR="${DROID_FSQ_EVAL_DIR:?launch with submit_eval.sh}"
PROJECT_ROOT="$(cd "${EVAL_DIR}/../../../../../../.." && pwd)"
export PROJECT_ROOT
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src/runtime_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SBD_GPU_GUARD_TERMINAL_ON_OOM=1
exec python -u "${EVAL_DIR}/src/run_eval.py" --config "${DROID_FSQ_EVAL_CONFIG:?missing frozen config}"
