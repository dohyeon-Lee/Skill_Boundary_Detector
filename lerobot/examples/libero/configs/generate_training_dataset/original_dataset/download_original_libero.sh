#!/usr/bin/env bash
# Download the ORIGINAL LIBERO dataset before converting it to our LeRobot format.
#
# Inputs / references:
#   - Downloader repo : https://github.com/huggingface/lerobot-libero
#   - Dataset suite   : LIBERO_ORIGINAL_DATASETS (default: libero_100)
#                     : libero_100 contains the original LIBERO-90 and LIBERO-10 demos.
#
# Output:
#   - Original data root : ${PROJECT_ROOT}/libero_original_dataset
#   - Expected contents  : original LIBERO HDF5 demos, typically under
#       ${PROJECT_ROOT}/libero_original_dataset/libero_90/
#       ${PROJECT_ROOT}/libero_original_dataset/libero_10/
#
# This does NOT write to ${PROJECT_ROOT}/libero_dataset. Conversion to the
# current LeRobot-v3.0 training format happens in a later step.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data2/dohyeon/SBD}"
DOWNLOAD_DIR="${LIBERO_ORIGINAL_DATASET_DIR:-${PROJECT_ROOT}/libero_original_dataset}"
TOOLS_DIR="${LIBERO_DOWNLOAD_TOOLS_DIR:-${PROJECT_ROOT}/tools}"
REPO_DIR="${LIBERO_LIBERO_REPO_DIR:-${TOOLS_DIR}/lerobot-libero}"
DATASETS="${LIBERO_ORIGINAL_DATASETS:-libero_100}"
USE_HUGGINGFACE="${LIBERO_USE_HUGGINGFACE:-1}"
UPDATE_REPO="${LIBERO_UPDATE_REPO:-0}"
INSTALL_REPO="${LIBERO_INSTALL_REPO:-1}"
HF_MAX_WORKERS="${LIBERO_HF_MAX_WORKERS:-2}"
HF_RETRIES="${LIBERO_HF_RETRIES:-20}"
HF_RETRY_SLEEP="${LIBERO_HF_RETRY_SLEEP:-30}"

# More stable defaults for large HDF5 downloads from Hugging Face. These can be
# overridden by setting the environment variables before running this script.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python3
fi

mkdir -p "${TOOLS_DIR}" "${DOWNLOAD_DIR}"

echo "Download original LIBERO dataset"
echo "  project_root : ${PROJECT_ROOT}"
echo "  downloader   : ${REPO_DIR}"
echo "  dataset      : ${DATASETS}"
echo "  output       : ${DOWNLOAD_DIR}"
echo "  python       : ${PYTHON_BIN}"
echo "  hf workers   : ${HF_MAX_WORKERS}"
echo "  hf retries   : ${HF_RETRIES}"
echo "  hf xet off   : ${HF_HUB_DISABLE_XET}"
echo "  hf timeout   : download=${HF_HUB_DOWNLOAD_TIMEOUT}s etag=${HF_HUB_ETAG_TIMEOUT}s"

if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "[1/3] Clone downloader repo"
  git clone https://github.com/huggingface/lerobot-libero "${REPO_DIR}"
elif [ "${UPDATE_REPO}" = "1" ]; then
  echo "[1/3] Update downloader repo"
  git -C "${REPO_DIR}" pull --ff-only
else
  echo "[1/3] Downloader repo already exists -> skip clone"
fi

if [ "${INSTALL_REPO}" = "1" ]; then
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    echo "[2/3] Install downloader package into current python env"
    "${PYTHON_BIN}" -m pip install -e "${REPO_DIR}"
  else
    echo "[2/3] Skip install: ${PYTHON_BIN} has no pip module."
    echo "      The downloader script adds its repo to PYTHONPATH via init_path, so this is usually OK."
    echo "      To force this behavior explicitly, rerun with LIBERO_INSTALL_REPO=0."
  fi
else
  echo "[2/3] Skip install (LIBERO_INSTALL_REPO=0)"
fi

HF_ARGS=()
if [ "${USE_HUGGINGFACE}" = "1" ]; then
  HF_ARGS=(--use-huggingface)
fi

echo "[3/3] Download dataset"
if [ "${USE_HUGGINGFACE}" = "1" ]; then
  # The HF mirror stores LIBERO-100 as libero_90/ + libero_10/, not libero_100/.
  # Use our helper so `libero_100` downloads the folders that actually exist.
  "${PYTHON_BIN}" "$(dirname "${BASH_SOURCE[0]}")/src/download_libero_hf.py" \
    --datasets "${DATASETS}" \
    --download-dir "${DOWNLOAD_DIR}" \
    --max-workers "${HF_MAX_WORKERS}" \
    --retries "${HF_RETRIES}" \
    --retry-sleep "${HF_RETRY_SLEEP}"
else
  "${PYTHON_BIN}" "${REPO_DIR}/benchmark_scripts/download_libero_datasets.py" \
    --datasets "${DATASETS}" \
    --download-dir "${DOWNLOAD_DIR}" \
    "${HF_ARGS[@]}"
fi

echo
echo "DONE"
echo "Original LIBERO data root:"
echo "  ${DOWNLOAD_DIR}"
echo
echo "Quick check:"
find "${DOWNLOAD_DIR}" -maxdepth 3 -type f \( -name "*.hdf5" -o -name "*.h5" \) | sort | head -20
echo
echo "HDF5/H5 file count:"
find "${DOWNLOAD_DIR}" -type f \( -name "*.hdf5" -o -name "*.h5" \) | wc -l
