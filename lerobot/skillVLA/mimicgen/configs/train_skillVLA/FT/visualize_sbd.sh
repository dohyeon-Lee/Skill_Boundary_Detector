#!/usr/bin/env bash
# Preview the SBD boundaries of the episodes selected for Coffee FT (CPU only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
exec "${PROJECT_ROOT}/.venv/bin/python" -u "${HERE}/src/visualize_sbd.py" "$@"
