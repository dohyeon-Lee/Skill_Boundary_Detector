#!/usr/bin/env bash
# Submit the Stage-1 terminator; it remains a separate training job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKILL_AUX_TRAIN_CONFIG="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_train_config.yaml}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../terminator/submit_train.sh" "$@"
