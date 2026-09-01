#!/usr/bin/env bash
# Build all three CALVIN long-horizon task-disjoint LeRobot datasets locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CALVIN_LONG_HORIZON_SPLIT_CONFIG:-${SCRIPT_DIR}/calvin_task_split_config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/../../../../../../../.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN=python3

echo "== CALVIN long-horizon three-way split =="
echo "  node   : ${SLURMD_NODENAME:-local}"
echo "  config : ${CONFIG_PATH}"

PYTHONDONTWRITEBYTECODE=1 CALVIN_LONG_HORIZON_SPLIT_CONFIG="${CONFIG_PATH}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_calvin_task_split.py"
