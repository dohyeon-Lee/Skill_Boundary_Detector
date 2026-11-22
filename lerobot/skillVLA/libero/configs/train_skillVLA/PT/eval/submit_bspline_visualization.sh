#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Same YAML/models as normal eval; separate one-episode Bspline diagnostics.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/pt_eval_config.yaml}"
export STAGE1_EVAL_CONFIG_RESOLVER="${SCRIPT_DIR}/src/bspline_visualization_config.py"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-BSvis}"
exec bash "${SCRIPT_DIR}/submit_eval.sh" "$@"
