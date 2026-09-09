#!/usr/bin/env bash
# Build the selected local DrawSVG groups into one canonical LeRobot v3 dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${DRAWSVG_CONFIG:-${SCRIPT_DIR}/drawsvg_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/drawsvg_dataset_config.py"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN="${BOOTSTRAP_PYTHON}"

ARGS=(
  --config "${CONFIG_PATH}"
  --source-root "${DRAWSVG_SOURCE_ROOT}"
  --output-root "${DRAWSVG_OUTPUT_ROOT}"
  --output-name "${DRAWSVG_OUTPUT_NAME}"
)
[ "${FORCE:-0}" = "1" ] && ARGS+=(--overwrite)
[ -n "${MAX_EPISODES:-}" ] && ARGS+=(--max-episodes "${MAX_EPISODES}")

echo "== DrawSVG -> LeRobot v3 =="
echo "  node   : ${SLURMD_NODENAME:-local}"
echo "  source : ${DRAWSVG_SOURCE_ROOT}"
echo "  groups : ${DRAWSVG_INCLUDE_GROUPS_JSON}"
echo "  output : ${DRAWSVG_OUTPUT_DIR}"

"${PYTHON_BIN}" "${DRAWSVG_CONVERT_SCRIPT}" "${ARGS[@]}"

# Match the other canonical dataset builders: exact dataset-wide quantiles for
# every non-video feature after the writer has finalized parquet/video metadata.
"${PYTHON_BIN}" "${DRAWSVG_ENSURE_STATS_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --root "${DRAWSVG_OUTPUT_ROOT}" \
  --dataset "${DRAWSVG_OUTPUT_NAME}" \
  --overwrite

echo "DONE -> ${DRAWSVG_OUTPUT_DIR}"
