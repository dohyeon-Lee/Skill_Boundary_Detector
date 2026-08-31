#!/usr/bin/env bash
# Convert one downloaded raw CALVIN split into a policy-ready LeRobot v3 dataset.
# All user-facing choices live in calvin_dataset_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CALVIN_CONFIG:-${SCRIPT_DIR}/calvin_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/calvin_dataset_config.py"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

RESOLVED_SETTINGS="$(PYTHONDONTWRITEBYTECODE=1 "${BOOTSTRAP_PYTHON}" \
  "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
eval "${RESOLVED_SETTINGS}"

if [ ! -d "${CALVIN_CONVERT_SOURCE_DIR}" ]; then
  echo "CALVIN source split not found: ${CALVIN_CONVERT_SOURCE_DIR}" >&2
  echo "Run ./download_calvin.sh first." >&2
  exit 1
fi
if [ -d "${CALVIN_CONVERT_OUTPUT_DIR}" ] && [ "${CALVIN_CONVERT_OVERWRITE}" != "true" ]; then
  echo "Converted output already exists: ${CALVIN_CONVERT_OUTPUT_DIR}" >&2
  echo "Set calvin_convert_overwrite: true or choose another output name." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN="${BOOTSTRAP_PYTHON}"

echo "== CALVIN LeRobot build =="
echo "  node     : ${SLURMD_NODENAME:-local}"
echo "  variant  : ${CALVIN_CONVERT_VARIANT}"
echo "  split    : ${CALVIN_CONVERT_SPLIT}"
echo "  mode     : ${CALVIN_CONVERT_MODE}"
echo "  task set : ${CALVIN_TASK_SPLIT}"
echo "  source   : ${CALVIN_CONVERT_SOURCE_DIR}"
echo "  output   : ${CALVIN_CONVERT_OUTPUT_DIR}"
echo "  action   : ${CALVIN_POLICY_ACTION}"
echo "  proprio  : ${CALVIN_POLICY_STATE}"
echo "  raw keep : ${CALVIN_PRESERVE_RAW_MODE}"

PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/src/convert_calvin_to_lerobot.py" --config "${CONFIG_PATH}"
