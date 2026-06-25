#!/usr/bin/env bash
# generate-mode DINO prep (GPU): compute every required (grid × camera) DINO directly on the
# SOURCE dataset's frames with a frozen DINOv3 pass, then identity-slice it into the work dir.
# Used when dino_base_dataset=generate (no precomputed base needed — handles merged datasets).
# Idempotent: precompute skips episodes whose npz already exist; re-running is safe.
#SBATCH --job-name=skillVLA_dinoprep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # train_skillVLA
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Freeze to the exact config resolved at submit time (env snapshot), else resolve now.
if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi
source "${SRC_DIR}/dino_prepare.sh"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA DINO-PREP (generate)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "Source    : ${RAW_DATASET_DIR}"
echo "DINO base : ${DINO_BASE_ROOT}   (computed from source frames)"
echo "Model     : ${DINO_MODEL_PATH}"
echo "Required  : ${DINO_REQUIRED}"
nvidia-smi || true

if [ ! -d "${DINO_MODEL_PATH}" ]; then
  echo "DINO model not found: ${DINO_MODEL_PATH}" >&2; exit 1
fi

# Per (grid, camera): precompute on the source frames → {source}_DINO/pg{grid}/{camera}, then
# identity-slice that into the work dir (base is keyed by the source's own episode index).
_dino_generate_and_slice () {
  local grid="$1" cam="$2"
  echo "── DINO grid=${grid} camera=${cam} ──"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/precompute_frame_dino_features.py \
      --dataset_dir      "${RAW_DATASET_DIR}" \
      --output_dir       "${DINO_BASE_ROOT}/pg${grid}" \
      --image_keys       "${cam}" \
      --image_model_name "${DINO_MODEL_PATH}" \
      --image_size       "${DINO_IMAGE_SIZE}" \
      --patch_grid       "${grid}" \
      --n_patch_raw      "${DINO_N_PATCH_RAW}" \
      --batch_size       "${DINO_BATCH_SIZE}" \
      --dtype            "${DINO_DTYPE}" \
      --device           cuda \
      --num_workers      1 \
      --worker_id        0
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/prepare_dino_for_skillvla.py" \
    --dataset_dir   "${RAW_DATASET_DIR}" \
    --base_dino_dir "${DINO_BASE_ROOT}/pg${grid}" \
    --dst_dir       "${DINO_ROOT}/pg${grid}" \
    --image_keys    "${cam}" \
    --mode          "${DINO_COPY_MODE}" \
    --identity
}
dino_for_each_required _dino_generate_and_slice

echo "DONE → ${DINO_ROOT}"
