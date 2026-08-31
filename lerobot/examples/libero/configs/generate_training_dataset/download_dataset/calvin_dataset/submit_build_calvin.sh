#!/usr/bin/env bash
# ONE-SHOT: ensure the selected CALVIN archive exists on this node, then submit
# the raw->LeRobot conversion as a Slurm job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CALVIN_CONFIG:-${SCRIPT_DIR}/calvin_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/calvin_dataset_config.py"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
RESOLVED_SETTINGS="$(PYTHONDONTWRITEBYTECODE=1 "${BOOTSTRAP_PYTHON}" \
  "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
eval "${RESOLVED_SETTINGS}"

echo "== ① download/verify selected raw variant on this node =="
CALVIN_CONFIG="${CONFIG_PATH}" CALVIN_ONLY="${CALVIN_CONVERT_VARIANT}" \
  "${SCRIPT_DIR}/download_calvin.sh"

if [ -d "${CALVIN_CONVERT_OUTPUT_DIR}" ] && [ "${CALVIN_CONVERT_OVERWRITE}" != "true" ]; then
  echo "Converted output already exists: ${CALVIN_CONVERT_OUTPUT_DIR}" >&2
  echo "Set calvin_convert_overwrite: true or choose another output name." >&2
  exit 1
fi

echo "== ② submit CALVIN -> LeRobot v3 build =="
cd "${SCRIPT_DIR}"
mkdir -p logs
SBATCH_ARGS=(
  --job-name=build_calvin
  --partition="${CALVIN_CONVERT_PARTITION}"
  --qos="${CALVIN_CONVERT_QOS}"
  --gres="${CALVIN_CONVERT_GRES}"
  --cpus-per-task="${CALVIN_CONVERT_CPUS_PER_TASK}"
  --mem="${CALVIN_CONVERT_MEM}"
  --time="${CALVIN_CONVERT_TIME}"
  --requeue
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
[ -n "${CALVIN_CONVERT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${CALVIN_CONVERT_NODELIST}")
[ -n "${CALVIN_CONVERT_EXCLUDE_NODES}" ] && \
  SBATCH_ARGS+=(--exclude="${CALVIN_CONVERT_EXCLUDE_NODES}")

WRAP="CALVIN_CONFIG=$(printf %q "${CONFIG_PATH}") ${SCRIPT_DIR}/build_calvin_dataset.sh"
echo "  source : ${CALVIN_CONVERT_SOURCE_DIR}"
echo "  output : ${CALVIN_CONVERT_OUTPUT_DIR}"
echo "  mode   : ${CALVIN_CONVERT_MODE} / ${CALVIN_TASK_SPLIT}"
echo "  slurm  : partition=${CALVIN_CONVERT_PARTITION} qos=${CALVIN_CONVERT_QOS}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
