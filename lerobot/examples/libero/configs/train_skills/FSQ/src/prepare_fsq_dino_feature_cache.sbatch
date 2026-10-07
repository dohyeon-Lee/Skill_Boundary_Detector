#!/usr/bin/env bash
# One-time GPU producer for frozen top-camera DINO tokens used by FSQ term22.
#SBATCH --job-name=FSQ_dino
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_cache/%x_%j.out
#SBATCH --error=logs_cache/%x_%j.err
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RAW_DATASET_DIR:?RAW_DATASET_DIR is required}"
: "${FSQ_FRAME_CACHE_DIR:?FSQ_FRAME_CACHE_DIR is required}"
: "${FSQ_DINO_FEATURE_CACHE_ROOT:?FSQ_DINO_FEATURE_CACHE_ROOT is required}"
: "${FSQ_DINO_FEATURE_CACHE_FINGERPRINT:?FSQ_DINO_FEATURE_CACHE_FINGERPRINT is required}"
: "${DINO_IMAGE_MODEL_PATH:?DINO_IMAGE_MODEL_PATH is required}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEROBOT_ROOT="${PROJECT_ROOT}/lerobot"
SHARED_VENV="$(readlink -f "${PROJECT_ROOT}/.venv")"
TRAIN_VENV="${SHARED_VENV}"
source "${SUBMIT_DIR}/src/fsq_node_local_venv.sh"
if [ -n "${FSQ_VENV_ARCHIVE:-}" ]; then
  if STAGED_VENV="$(fsq_stage_venv_on_node "${FSQ_VENV_ARCHIVE}" "${SHARED_VENV}")"; then
    TRAIN_VENV="${STAGED_VENV}"
  fi
fi
PYTHON="${TRAIN_VENV}/bin/python"
export VIRTUAL_ENV="${TRAIN_VENV}"
export PATH="${TRAIN_VENV}/bin:${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export LD_LIBRARY_PATH=""

cd "${LEROBOT_ROOT}"
echo "START FSQ DINO FEATURE CACHE"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Dataset    : ${RAW_DATASET_DIR}"
echo "RGB cache  : ${FSQ_FRAME_CACHE_DIR}"
echo "Cache root : ${FSQ_DINO_FEATURE_CACHE_ROOT}"
echo "Fingerprint: ${FSQ_DINO_FEATURE_CACHE_FINGERPRINT}"
echo "Batch size : ${FSQ_DINO_FEATURE_BATCH_SIZE:-256}"
nvidia-smi

"${PYTHON}" -u examples/libero/fsq_dino_feature_cache.py build \
  --raw-dataset-dir "${RAW_DATASET_DIR}" \
  --rgb-cache-dir "${FSQ_FRAME_CACHE_DIR}" \
  --model-path "${DINO_IMAGE_MODEL_PATH}" \
  --image-size "${FSQ_DINO_IMAGE_SIZE}" \
  --cache-root "${FSQ_DINO_FEATURE_CACHE_ROOT}" \
  --expected-fingerprint "${FSQ_DINO_FEATURE_CACHE_FINGERPRINT}" \
  --batch-size "${FSQ_DINO_FEATURE_BATCH_SIZE:-256}"

"${PYTHON}" -u examples/libero/fsq_dino_feature_cache.py validate \
  --raw-dataset-dir "${RAW_DATASET_DIR}" \
  --model-path "${DINO_IMAGE_MODEL_PATH}" \
  --image-size "${FSQ_DINO_IMAGE_SIZE}" \
  --cache-root "${FSQ_DINO_FEATURE_CACHE_ROOT}"

echo "DONE FSQ DINO FEATURE CACHE"
