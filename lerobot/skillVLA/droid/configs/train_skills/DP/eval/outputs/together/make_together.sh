#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../../../remote.sh"
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${EVAL_DIR}/../../../../../../.." && pwd)"
exec "${PROJECT_ROOT}/.venv/bin/python" "${EVAL_DIR}/src/together_report.py" \
  --config "${SCRIPT_DIR}/together_config.yaml" "$@"
