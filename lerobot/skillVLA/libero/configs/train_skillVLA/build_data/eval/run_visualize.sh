#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
exec "${PROJECT_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/src/visualize_skills.py" \
  --config "${1:-${SCRIPT_DIR}/visualize_config.yaml}"
