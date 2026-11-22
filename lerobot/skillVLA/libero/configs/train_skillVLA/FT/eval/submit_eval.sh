#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Submit closed-loop evaluation of NewTask FT models. This is the Stage-1 eval engine
# (../../PT/eval/src) run with this folder's YAML; logs/ and outputs/ are written here.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/ft_eval_config.yaml}"
export STAGE1_EVAL_WORK_DIR="${SCRIPT_DIR}"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-FTeval}"
exec "${SCRIPT_DIR}/../../PT/eval/submit_eval.sh" "$@"
