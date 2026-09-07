#!/usr/bin/env bash
# Resolve FT-specific paths, then use the maintained Stage-2 submission stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE2_EVAL_CONFIG="${FT_EVAL_CONFIG:-${SCRIPT_DIR}/ft_eval_config.yaml}"
export STAGE2_EVAL_CONFIG_RESOLVER="${SCRIPT_DIR}/src/ft_eval_config.py"
export STAGE2_EVAL_JOB_NAME="${STAGE2_EVAL_JOB_NAME:-FTeval}"
export STAGE2_EVAL_DISPLAY_NAME="${STAGE2_EVAL_DISPLAY_NAME:-Stage-2 FT eval}"
export STAGE2_EVAL_LOG_DIR="${STAGE2_EVAL_LOG_DIR:-${SCRIPT_DIR}/logs}"
export STAGE2_EVAL_VENV_LABEL="${STAGE2_EVAL_VENV_LABEL:-FT eval venv}"

exec "${SCRIPT_DIR}/../stage2_eval/submit_eval.sh"
