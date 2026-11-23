#!/usr/bin/env bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
EVAL_DIR="${LIBERO_DP_EVAL_DIR:?launch with submit_eval.sh}"
PROJECT_ROOT="$(cd "${EVAL_DIR}/../../../../../../.." && pwd)"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${EVAL_DIR}/../../../src/runtime_env.sh"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u "${EVAL_DIR}/src/run_eval.py" --config "${LIBERO_DP_EVAL_CONFIG:?missing frozen config}"
