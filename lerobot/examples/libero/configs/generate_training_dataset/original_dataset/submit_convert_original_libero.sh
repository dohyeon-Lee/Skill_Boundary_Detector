#!/usr/bin/env bash
# Submit original LIBERO HDF5 -> LeRobot v3 conversion.
#
# Inputs:
#   HDF5 demos : {convert_source_root}/{convert_suite}/*.hdf5
# Reference model:
#   none
# Outputs:
#   dataset    : {convert_output_root}/{convert_output_name}
#
# Config:
#   ./original_dataset_config.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${ORIGINAL_DATASET_CONFIG:-${SCRIPT_DIR}/original_dataset_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
CONFIG_PY="${SRC_DIR}/original_dataset_config.py"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

if [ ! -d "${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}" ]; then
  echo "Source HDF5 folder not found: ${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}" >&2
  exit 1
fi
if [ -d "${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}" ] && [ "${CONVERT_OVERWRITE}" != "true" ]; then
  echo "Output already exists: ${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}" >&2
  echo "Set convert_overwrite: true or choose a new convert_output_name." >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${CONVERT_PARTITION}"
  --qos="${CONVERT_QOS}"
  --gres="${CONVERT_GRES}"
  --cpus-per-task="${CONVERT_CPUS_PER_TASK}"
  --mem="${CONVERT_MEM}"
  --time="${CONVERT_TIME}"
)
[ -n "${CONVERT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${CONVERT_NODELIST}")
[ -n "${CONVERT_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${CONVERT_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit original LIBERO conversion"
echo "  source : ${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}"
echo "  output : ${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}"
echo "  codec  : ${CONVERT_VCODEC}"
echo "  slurm  : partition=${CONVERT_PARTITION} nodelist=${CONVERT_NODELIST:-<none>} exclude=${CONVERT_EXCLUDE_NODES:-<none>}"

ORIGINAL_DATASET_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/convert_original_libero.sbatch"
