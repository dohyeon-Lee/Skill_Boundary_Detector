#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing training environment: ${PYTHON}. Run bash setup_env.sh first." >&2
  exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/visualize_training_dataset.py" "$@"
