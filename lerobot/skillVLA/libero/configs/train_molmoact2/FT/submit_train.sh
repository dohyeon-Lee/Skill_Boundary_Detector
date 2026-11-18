#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../PT/src/submit.sh" "${SCRIPT_DIR}" "${MOLMOACT2_TRAIN_CONFIG:-${SCRIPT_DIR}/molmoact2_FT_config.yaml}" "$@"
