#!/usr/bin/env bash
# Same YAML/models as normal eval; separate one-episode Bspline diagnostics.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_eval_config.yaml}"
export STAGE1_EVAL_CONFIG_RESOLVER="${SCRIPT_DIR}/src/bspline_visualization_config.py"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-BSvis}"
exec bash "${SCRIPT_DIR}/submit_eval.sh" "$@"
