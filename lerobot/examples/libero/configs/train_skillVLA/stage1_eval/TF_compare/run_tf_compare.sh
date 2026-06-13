#!/usr/bin/env bash
# Stage-1 teacher-forced comparison — run teacher_forced.py for every target in
# tf_compare_config.yaml (cached to outputs/{name}/metrics.json) then render outputs/summary.md.
# Usage: ./run_tf_compare.sh [--force]   (needs a GPU; use submit_tf_compare.sh to queue it)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/tf_compare_config.py" --shell)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

FORCE="${1:-}"
python "${SCRIPT_DIR}/src/run_targets.py" ${FORCE:+--force}
python "${SCRIPT_DIR}/src/report.py"
