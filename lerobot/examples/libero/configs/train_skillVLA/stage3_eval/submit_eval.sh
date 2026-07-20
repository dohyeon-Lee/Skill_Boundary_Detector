#!/usr/bin/env bash
# Stage-3 eval frontend. The rollout/fan-out engine is shared with stage2_eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_DIR="${SCRIPT_DIR}/../stage2_eval"

export SKILLVLA_EVAL_FRONTEND_DIR="${SCRIPT_DIR}"
export SKILLVLA_EVAL_RUNNER_DIR="${RUNNER_DIR}"
export SKILLVLA_EVAL_CONFIG="${STAGE3_EVAL_CONFIG:-${SCRIPT_DIR}/stage3_eval_config.yaml}"
export SKILLVLA_EVAL_CONFIG_EMITTER="${SCRIPT_DIR}/src/stage3_eval_config.py"
export SKILLVLA_EVAL_JOB_NAME="S3eval"
export SKILLVLA_EVAL_TRAIN_STAGE="stage3"

exec "${RUNNER_DIR}/submit_eval.sh"
