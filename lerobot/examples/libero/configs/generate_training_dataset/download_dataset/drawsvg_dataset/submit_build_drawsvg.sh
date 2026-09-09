#!/usr/bin/env bash
# Validate the local DrawSVG inputs, freeze config, then submit one conversion job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${DRAWSVG_CONFIG:-${SCRIPT_DIR}/drawsvg_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/drawsvg_dataset_config.py"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do
  PARENT="$(dirname "${CONFIG_LIB}")"
  if [ "${PARENT}" = "${CONFIG_LIB}" ]; then
    echo "snapshot_config.sh not found above ${CONFIG_PATH}" >&2
    exit 1
  fi
  CONFIG_LIB="${PARENT}"
done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN="${BOOTSTRAP_PYTHON}"

if [ -d "${DRAWSVG_OUTPUT_DIR}" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "Output already exists: ${DRAWSVG_OUTPUT_DIR}" >&2
  echo "Set FORCE=1 only if this exact output should be replaced." >&2
  exit 1
fi

echo "== ① validate selected local DrawSVG groups =="
"${PYTHON_BIN}" "${DRAWSVG_CONVERT_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --source-root "${DRAWSVG_SOURCE_ROOT}" \
  --output-root "${DRAWSVG_OUTPUT_ROOT}" \
  --output-name "${DRAWSVG_OUTPUT_NAME}" \
  --dry-run

echo "== ② submit DrawSVG -> LeRobot v3 build =="
cd "${SCRIPT_DIR}"
mkdir -p logs
SBATCH_ARGS=(
  --job-name=build_drawsvg
  --partition="${DRAWSVG_CONVERT_PARTITION}"
  --qos="${DRAWSVG_CONVERT_QOS}"
  --gres="${DRAWSVG_CONVERT_GRES}"
  --cpus-per-task="${DRAWSVG_CONVERT_CPUS_PER_TASK}"
  --mem="${DRAWSVG_CONVERT_MEM}"
  --time="${DRAWSVG_CONVERT_TIME}"
  --requeue
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
[ -n "${DRAWSVG_CONVERT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DRAWSVG_CONVERT_NODELIST}")
[ -n "${DRAWSVG_CONVERT_EXCLUDE_NODES}" ] && \
  SBATCH_ARGS+=(--exclude="${DRAWSVG_CONVERT_EXCLUDE_NODES}")

WRAP="DRAWSVG_CONFIG=$(printf %q "${CONFIG_PATH}")"
[ "${FORCE:-0}" = "1" ] && WRAP="FORCE=1 ${WRAP}"
WRAP="${WRAP} ${SCRIPT_DIR}/build_drawsvg_dataset.sh"

echo "  groups : ${DRAWSVG_INCLUDE_GROUPS_JSON}"
echo "  output : ${DRAWSVG_OUTPUT_DIR}"
echo "  slurm  : partition=${DRAWSVG_CONVERT_PARTITION} qos=${DRAWSVG_CONVERT_QOS}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
