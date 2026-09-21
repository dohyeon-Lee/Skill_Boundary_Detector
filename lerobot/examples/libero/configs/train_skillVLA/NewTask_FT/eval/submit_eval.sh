#!/usr/bin/env bash
# Submit closed-loop evaluation of NewTask FT models. This is the Stage-1 eval engine
# (../../stage1_eval/src) run with this folder's YAML; logs/ and outputs/ are written here.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/ft_eval_config.yaml}"
export STAGE1_EVAL_WORK_DIR="${SCRIPT_DIR}"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-FTeval}"
exec "${SCRIPT_DIR}/../../stage1_eval/submit_eval.sh" "$@"
