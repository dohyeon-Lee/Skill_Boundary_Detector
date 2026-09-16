#!/usr/bin/env bash
# Submit the Stage-1 VSA using its component YAML and shared Stage-1 defaults.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_TRAIN_CONFIG="${STAGE1_TRAIN_CONFIG:-${SCRIPT_DIR}/vsa_train_config.yaml}"
export STAGE1_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../stage1/submit_train.sh" "$@"
