#!/usr/bin/env bash
# ONE-SHOT: download original LIBERO HDF5 on THIS node (login → internet
# guaranteed), then submit HDF5→LeRobot-v3 conversion + stats as a Slurm job.
#
# Usage (from this folder):
#   ./submit_build_original.sh
#   CONVERT_SUITE=libero_10 ./submit_build_original.sh
#   CONVERT_OVERWRITE=true ./submit_build_original.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ORIGINAL_DATASET_CONFIG:-${SCRIPT_DIR}/original_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/original_dataset_config.py"

# Freeze one config for both download and the queued build.
_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

echo "== ① download (this node) =="
ORIGINAL_DATASET_CONFIG="${CONFIG_PATH}" "${SCRIPT_DIR}/download_original_libero.sh"

SOURCE_DIR="${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}"
OUTPUT_DIR="${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}"
if [ ! -d "${SOURCE_DIR}" ]; then
  echo "Source HDF5 folder not found after download: ${SOURCE_DIR}" >&2
  exit 1
fi
if [ -d "${OUTPUT_DIR}" ] && [ "${CONVERT_OVERWRITE}" != "true" ]; then
  echo "Output already exists: ${OUTPUT_DIR}" >&2
  echo "Set convert_overwrite: true or choose a new convert_output_name." >&2
  exit 1
fi

echo "== ② submit build job (HDF5→v3 + stats) =="
cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=build_original
  --partition="${CONVERT_PARTITION}"
  --qos="${CONVERT_QOS}"
  --gres="${CONVERT_GRES}"
  --cpus-per-task="${CONVERT_CPUS_PER_TASK}"
  --mem="${CONVERT_MEM}"
  --time="${CONVERT_TIME}"
  --requeue
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
[ -n "${CONVERT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${CONVERT_NODELIST}")
[ -n "${CONVERT_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${CONVERT_EXCLUDE_NODES}")

WRAP="ORIGINAL_DATASET_CONFIG=$(printf %q "${CONFIG_PATH}") \
${SCRIPT_DIR}/build_original_dataset.sh"

echo "  suite  : ${CONVERT_SUITE}"
echo "  source : ${SOURCE_DIR}"
echo "  output : ${OUTPUT_DIR}"
echo "  schema : ${CONVERT_SCHEMA_REFERENCE}"
echo "  slurm  : partition=${CONVERT_PARTITION} nodelist=${CONVERT_NODELIST:-<none>} exclude=${CONVERT_EXCLUDE_NODES:-<none>}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
