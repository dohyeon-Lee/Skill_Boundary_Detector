#!/usr/bin/env bash
# Per-block term-2 (inter-group gradient alignment) map for a cycle-PT checkpoint.
# Usage: MODEL=<run_dir> CKPT=020000 sbatch term2.sbatch   (MODEL under pi05_cycle_PT)
#SBATCH --job-name=pi05_term2
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../../../.." && pwd)"
LEROBOT_ROOT="${PROJECT_ROOT}/lerobot"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
PI05_TOKENIZER_PATH="${PROJECT_ROOT}/models/paligemma-3b-pt-224-tokenizer"

MODEL="${MODEL:-PTcyc_libero_90_full_full_pi05_batch16_g8p250_constlr}"
CKPT="${CKPT:-020000}"
PT_ROOT="${PROJECT_ROOT}/outputs_filtered/pi05_cycle_PT"
CKPT_DIR="${PT_ROOT}/${MODEL}/checkpoints/${CKPT}/pretrained_model"
GROUPS_JSON="${PT_ROOT}/${MODEL}/groups.json"
DATASET_ROOT="${PROJECT_ROOT}/dataset_filtered/libero_90_full_full"
OUT_DIR="${SUBMIT_DIR}/outputs_term2/${MODEL}_${CKPT}"

echo "checkpoint : ${CKPT_DIR}"
echo "groups     : ${GROUPS_JSON}"
nvidia-smi || true

PYTHONPATH="${LEROBOT_ROOT}/src" python \
  "${ROOT_DIR}/src/measure_term2_blocks.py" \
  --checkpoint "${CKPT_DIR}" \
  --groups_json "${GROUPS_JSON}" \
  --dataset_root "${DATASET_ROOT}" \
  --tokenizer_path "${PI05_TOKENIZER_PATH}" \
  --out_dir "${OUT_DIR}"

echo "DONE -> ${OUT_DIR}"
