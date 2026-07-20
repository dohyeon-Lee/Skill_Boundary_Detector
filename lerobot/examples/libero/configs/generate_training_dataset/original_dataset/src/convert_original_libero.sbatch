#!/usr/bin/env bash
# Convert original LIBERO HDF5 into local LeRobot v3 format.
#
# Inputs:
#   HDF5 demos : {convert_source_root}/{convert_suite}/*.hdf5
# Reference model:
#   none
# Outputs:
#   dataset    : {convert_output_root}/{convert_output_name}
#
# Config:
#   ../original_dataset_config.yaml
#SBATCH --job-name=convert_libero
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${ORIGINAL_DATASET_CONFIG:-${SUBMIT_DIR}/original_dataset_config.yaml}"
CONFIG_PY="${SRC_DIR}/original_dataset_config.py"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

mkdir -p "${SUBMIT_DIR}/logs"
cd "${SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN=python3

ARGS=(
  --config "${CONFIG_PATH}"
  --suite "${CONVERT_SUITE}"
  --source-root "${CONVERT_SOURCE_ROOT}"
  --output-root "${CONVERT_OUTPUT_ROOT}"
  --output-name "${CONVERT_OUTPUT_NAME}"
  --image-size "${CONVERT_IMAGE_SIZE}"
  --image-writer-threads "${CONVERT_IMAGE_WRITER_THREADS}"
  --image-writer-processes "${CONVERT_IMAGE_WRITER_PROCESSES}"
  --vcodec "${CONVERT_VCODEC}"
  --batch-encoding-size "${CONVERT_BATCH_ENCODING_SIZE}"
  --encoder-queue-maxsize "${CONVERT_ENCODER_QUEUE_MAXSIZE}"
  --schema "${CONVERT_SCHEMA}"
)

[ "${CONVERT_OVERWRITE}" = "true" ] && ARGS+=(--overwrite)
[ "${CONVERT_STREAMING_ENCODING}" = "true" ] && ARGS+=(--streaming-encoding)
[ -n "${CONVERT_ENCODER_THREADS}" ] && ARGS+=(--encoder-threads "${CONVERT_ENCODER_THREADS}")
[ -n "${CONVERT_MAX_TASKS}" ] && ARGS+=(--max-tasks "${CONVERT_MAX_TASKS}")
[ -n "${CONVERT_MAX_EPISODES_PER_TASK}" ] && ARGS+=(--max-episodes-per-task "${CONVERT_MAX_EPISODES_PER_TASK}")

echo "START original LIBERO conversion"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Suite      : ${CONVERT_SUITE}"
echo "Source     : ${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}"
echo "Output     : ${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}"
echo "Schema     : ${CONVERT_SCHEMA}"
echo "Codec      : ${CONVERT_VCODEC}"
echo "Overwrite  : ${CONVERT_OVERWRITE}"
nvidia-smi || true

"${PYTHON_BIN}" "${CONVERT_SCRIPT}" "${ARGS[@]}"

echo "DONE -> ${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}"
