#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "Missing ${PYTHON}; run setup_env.sh first." >&2
  exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/visualize_training_dataset.py" "$@"
