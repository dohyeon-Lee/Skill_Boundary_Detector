#!/usr/bin/env bash
# Submit Predictor training (scalar: one model; mode list: share one GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKILL_AUX_TRAIN_CONFIG="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/predictor_train_config.yaml}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../Terminator/src/submit_auxiliary.sh" "$@"
