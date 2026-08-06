#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/stack_eval_outputs_config.yaml}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

[ -x "${PYTHON}" ] || PYTHON=python3
exec "${PYTHON}" "${SCRIPT_DIR}/src/stack_eval_outputs.py" --config "${CONFIG_PATH}" "${@:2}"
