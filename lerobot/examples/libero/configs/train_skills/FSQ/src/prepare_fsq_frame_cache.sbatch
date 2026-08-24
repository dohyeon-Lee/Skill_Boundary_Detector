#!/usr/bin/env bash
# CPU-only producer for the exact lossless-zstd RGB cache used by visual FSQ.
# Resource values are supplied by submit_train_fsq.sh.
#SBATCH --job-name=FSQ_frames
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs_cache/%x_%j.out
#SBATCH --error=logs_cache/%x_%j.err
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RAW_DATASET_DIR:?RAW_DATASET_DIR is required}"
: "${FSQ_FRAME_CACHE_ROOT:?FSQ_FRAME_CACHE_ROOT is required}"
: "${FSQ_FRAME_CACHE_FINGERPRINT:?FSQ_FRAME_CACHE_FINGERPRINT is required}"

LEROBOT_ROOT="${PROJECT_ROOT}/lerobot"
SHARED_VENV="$(readlink -f "${PROJECT_ROOT}/.venv")"
PYTHON="${SHARED_VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "FSQ frame cache Python is not executable: ${PYTHON}" >&2
  exit 1
fi

cd "${LEROBOT_ROOT}"
echo "START FSQ FRAME CACHE"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Dataset    : ${RAW_DATASET_DIR}"
echo "Cache root : ${FSQ_FRAME_CACHE_ROOT}"
echo "Fingerprint: ${FSQ_FRAME_CACHE_FINGERPRINT}"
echo "Workers    : ${FSQ_FRAME_CACHE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

LEGACY_CACHE_ROOT="$(dirname "${FSQ_FRAME_CACHE_ROOT}")/rgb_uint8_v1"
echo "Legacy RGB : ${LEGACY_CACHE_ROOT} (used only when complete)"

"${PYTHON}" -u examples/libero/fsq_frame_cache.py build \
  --raw-dataset-dir "${RAW_DATASET_DIR}" \
  --cache-root "${FSQ_FRAME_CACHE_ROOT}" \
  --expected-fingerprint "${FSQ_FRAME_CACHE_FINGERPRINT}" \
  --workers "${FSQ_FRAME_CACHE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}" \
  --decoder-threads "${FSQ_FRAME_CACHE_DECODER_THREADS:-1}" \
  --legacy-cache-root "${LEGACY_CACHE_ROOT}"

"${PYTHON}" -u examples/libero/fsq_frame_cache.py validate \
  --raw-dataset-dir "${RAW_DATASET_DIR}" \
  --cache-root "${FSQ_FRAME_CACHE_ROOT}"

echo "DONE FSQ FRAME CACHE"
