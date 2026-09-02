#!/usr/bin/env bash
# Compare the original and predictor-relabeled SkillVLA labels and build a
# self-contained HTML report. This is a lightweight local CPU analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${RELABEL_EVAL_CONFIG:-${SCRIPT_DIR}/relabel_eval_config.yaml}"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON=python3
fi

"${PYTHON}" "${SCRIPT_DIR}/src/eval_relabel.py" --config "${CONFIG_PATH}"
