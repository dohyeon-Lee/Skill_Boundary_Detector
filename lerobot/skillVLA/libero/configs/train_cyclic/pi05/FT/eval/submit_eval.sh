#!/usr/bin/env bash
# Reuse current paired pi05 evaluation, with cyclic-local configuration and outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_EVAL_CONFIG="${PI05_EVAL_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}"
export PI05_EVAL_WORK_DIR="${SCRIPT_DIR}"
export PI05_EVAL_JOB_NAME="${PI05_EVAL_JOB_NAME:-cyclicFTretention}"
exec bash "${SCRIPT_DIR}/../../../../train_pi05/PT/eval/submit_eval.sh" "$@"
