#!/usr/bin/env bash
# Run the shared PT evaluation engine with FT settings, logs and outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_EVAL_CONFIG="${PI05_EVAL_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}"
export PI05_EVAL_WORK_DIR="${SCRIPT_DIR}"
export PI05_EVAL_JOB_NAME="${PI05_EVAL_JOB_NAME:-pi05FTeval}"
exec "${SCRIPT_DIR}/../../PT/eval/submit_eval.sh" "$@"
