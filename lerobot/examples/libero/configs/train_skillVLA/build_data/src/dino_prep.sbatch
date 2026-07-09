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
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

# Phase, set by the submitter (backward compat: default "all" = the old single-GPU behaviour):
#   all        — precompute EVERY episode-shard (worker 0/1) then slice. Single GPU.
#   precompute — compute only this array task's episode shard (worker SLURM_ARRAY_TASK_ID / N).
#   slice      — identity-slice once, after all precompute shards finished (a dependent single job).
# Episodes are sharded by (i % num_workers == worker_id) inside precompute_frame_dino_features.py,
# so slice MUST wait for every shard — a per-worker slice would make dangling symlinks for episodes
# other workers haven't computed yet, which the downstream skillset stage would then fail to read.
DINO_PREP_PHASE="${DINO_PREP_PHASE:-all}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"
NW="${DINO_GENERATE_WORKERS:-1}"

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
echo "Phase     : ${DINO_PREP_PHASE}   worker ${WORKER_ID}/${NW}"
nvidia-smi || true

if [ ! -d "${DINO_MODEL_PATH}" ]; then
  echo "DINO model not found: ${DINO_MODEL_PATH}" >&2; exit 1
fi

# Precompute this worker's episode shard for one (grid, camera) → {source}_DINO/pg{grid}/{camera}.
# Sharding is by episode index (i % NW == WORKER_ID); NW=1/WORKER_ID=0 computes everything.
_dino_generate () {
  local grid="$1" cam="$2"
  echo "── DINO grid=${grid} camera=${cam}  (worker ${WORKER_ID}/${NW}) ──"
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
      --num_workers      "${NW}" \
      --worker_id        "${WORKER_ID}"
}

case "${DINO_PREP_PHASE}" in
  precompute)
    dino_for_each_required _dino_generate
    echo "DONE precompute (worker ${WORKER_ID}/${NW})" ;;
  slice)
    # identity-slice base → work dir (all shards are present now; dino_prepare_slice adds --identity
    # in generate mode). Pure symlink/copy, but must run after the whole precompute array.
    dino_prepare_slice
    echo "DONE slice → ${DINO_ROOT}" ;;
  all|*)
    dino_for_each_required _dino_generate
    dino_prepare_slice
    echo "DONE → ${DINO_ROOT}" ;;
esac
