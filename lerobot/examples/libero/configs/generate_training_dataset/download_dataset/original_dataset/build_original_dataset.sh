#!/usr/bin/env bash
# Build the original LIBERO LeRobot-v3 dataset from staged HDF5 demos:
#   ② HDF5→canonical v3  ③ exact non-video quantile stats
# Download is NOT done here — run ./download_original_libero.sh (or
# ./submit_build_original.sh) first.
#
# Usage:
#   ./build_original_dataset.sh
#   CONVERT_SUITE=libero_10 ./build_original_dataset.sh
#   CONVERT_OVERWRITE=true ./build_original_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ORIGINAL_DATASET_CONFIG:-${SCRIPT_DIR}/original_dataset_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/original_dataset_config.py"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

SOURCE_DIR="${CONVERT_SOURCE_ROOT}/${CONVERT_SUITE}"
OUTPUT_DIR="${CONVERT_OUTPUT_ROOT}/${CONVERT_OUTPUT_NAME}"
if [ ! -d "${SOURCE_DIR}" ]; then
  echo "Source HDF5 folder not found: ${SOURCE_DIR}" >&2
  echo "Run ./download_original_libero.sh first." >&2
  exit 1
fi
if [ -d "${OUTPUT_DIR}" ] && [ "${CONVERT_OVERWRITE}" != "true" ]; then
  echo "Output already exists: ${OUTPUT_DIR}" >&2
  echo "Set convert_overwrite: true or choose a new convert_output_name." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN="${BOOTSTRAP_PYTHON}"

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
  --schema-reference "${CONVERT_SCHEMA_REFERENCE}"
)

[ "${CONVERT_OVERWRITE}" = "true" ] && ARGS+=(--overwrite)
[ "${CONVERT_STREAMING_ENCODING}" = "true" ] && ARGS+=(--streaming-encoding)
[ -n "${CONVERT_ENCODER_THREADS}" ] && ARGS+=(--encoder-threads "${CONVERT_ENCODER_THREADS}")
[ -n "${CONVERT_MAX_TASKS}" ] && ARGS+=(--max-tasks "${CONVERT_MAX_TASKS}")
[ -n "${CONVERT_MAX_EPISODES_PER_TASK}" ] && \
  ARGS+=(--max-episodes-per-task "${CONVERT_MAX_EPISODES_PER_TASK}")

echo "== Original LIBERO build =="
echo "  node   : ${SLURMD_NODENAME:-local}"
echo "  suite  : ${CONVERT_SUITE}"
echo "  source : ${SOURCE_DIR}"
echo "  output : ${OUTPUT_DIR}"
echo "  schema : ${CONVERT_SCHEMA_REFERENCE}"
echo "  codec  : ${CONVERT_VCODEC}"
nvidia-smi || true

# Submitted builds retain the existing bad-GPU failover behavior. Direct local
# builds may run without a Slurm allocation.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
  require_cuda_or_requeue "${PYTHON_BIN}"
fi

"${PYTHON_BIN}" "${CONVERT_SCRIPT}" "${ARGS[@]}"

# The writer creates stats while saving. Recompute exact dataset-wide non-video
# quantiles once so every original-LIBERO build has the same final contract.
"${PYTHON_BIN}" "${ENSURE_STATS_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --root "${CONVERT_OUTPUT_ROOT}" \
  --dataset "${CONVERT_OUTPUT_NAME}" \
  --overwrite

echo "DONE -> ${OUTPUT_DIR}"
