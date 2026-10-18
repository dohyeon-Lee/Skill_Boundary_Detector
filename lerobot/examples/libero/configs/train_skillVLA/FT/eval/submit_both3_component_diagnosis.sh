#!/usr/bin/env bash
# Dedicated entry point; leaves ft_eval_config.yaml and its normal submit unchanged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${SCRIPT_DIR}/both3_50k_component_diagnosis.yaml"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-B3diag}"
exec bash "${SCRIPT_DIR}/submit_eval.sh" "$@"
