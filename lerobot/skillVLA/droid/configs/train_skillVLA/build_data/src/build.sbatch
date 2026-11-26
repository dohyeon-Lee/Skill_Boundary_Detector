#!/usr/bin/env bash
set -euo pipefail
DIR="${DROID_SKILLVLA_BUILD_DIR:?missing build directory}"
PROJECT_ROOT="$(cd "${DIR}/../../../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="$(dirname "${DROID_SKILLVLA_BUILD_CONFIG}")/global_config.yaml"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export SBD_GPU_GUARD_TERMINAL_ON_OOM=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec "${PROJECT_ROOT}/.venv/bin/python" -u "${DIR}/src/build_dataset.py" --config "${DROID_SKILLVLA_BUILD_CONFIG}" --run
