#!/usr/bin/env bash
# Stage-2 VLM attention heatmap (language → 3rd-person image). Loads a trained Stage-2 checkpoint,
# runs ONLY the VLM prefix on skill-start frames, and renders the per-word attention over the image.
#SBATCH --job-name=S2_attn_viz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # attn_viz
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${ATTN_VIZ_CONFIG:-${SUBMIT_DIR}/attn_viz_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/attn_viz_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

echo "START Stage-2 VLM attn_viz"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "policy  : ${POLICY_PATH}"
echo "dataset : ${DATASET_DIR}"
echo "output  : ${OUTPUT_DIR}   (n_samples=${N_SAMPLES}, layers=${ATTN_LAYERS})"
nvidia-smi || true

if [ ! -d "${POLICY_PATH}" ]; then echo "Policy not found: ${POLICY_PATH}" >&2; exit 1; fi
if [ ! -d "${DATASET_DIR}" ]; then echo "Dataset not found: ${DATASET_DIR}" >&2; exit 1; fi

PYTHONPATH="${LEROBOT_ROOT}/src" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_attn_viz.py" \
    --policy_path "${POLICY_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --n_samples   "${N_SAMPLES}" \
    --layers      "${ATTN_LAYERS}" \
    --weighting   "${ATTN_WEIGHTING}" \
    --seed        "${SEED}" \
    --device      cuda

echo "DONE → ${OUTPUT_DIR}/index_lang2img.html , index_img2lang.html , index_skillq.html"
