#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/src/submit.sh" "${SCRIPT_DIR}" "${GROOT_TRAIN_CONFIG:-${SCRIPT_DIR}/groot_PT_config.yaml}" "$@"
