#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Run the shared PT evaluation engine with FT settings, logs and outputs.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_EVAL_CONFIG="${PI05_EVAL_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}"
export PI05_EVAL_WORK_DIR="${SCRIPT_DIR}"
export PI05_EVAL_JOB_NAME="${PI05_EVAL_JOB_NAME:-pi05FTeval}"
exec "${SCRIPT_DIR}/../../PT/eval/submit_eval.sh" "$@"
