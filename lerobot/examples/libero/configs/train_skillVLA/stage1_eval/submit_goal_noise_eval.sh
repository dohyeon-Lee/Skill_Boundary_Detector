#!/usr/bin/env bash
# Submit paired raw-vs-normalized goal-noise evaluation through the shared engine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${STAGE1_GOAL_NOISE_CONFIG:-${SCRIPT_DIR}/stage1_goal_noise_eval_config.yaml}"
export STAGE1_EVAL_CONFIG_RESOLVER="${SCRIPT_DIR}/src/stage1_goal_noise_eval_config.py"
export STAGE1_EVAL_JOB_NAME="${STAGE1_EVAL_JOB_NAME:-S1goalnoise}"

exec "${SCRIPT_DIR}/submit_eval.sh"
