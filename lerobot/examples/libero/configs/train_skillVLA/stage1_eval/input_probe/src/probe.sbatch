#!/usr/bin/env bash
# Stage-1 INPUT-influence probe (offline, no simulator): perturb ONE input (state / skill / progress)
# and measure the action-chunk change vs the image-zero reference. See src/input_probe.py.
#SBATCH --job-name=input_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${INPUT_PROBE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # input_probe
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${INPUT_PROBE_CONFIG:-${SUBMIT_DIR}/input_probe_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/input_probe_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

echo "START INPUT PROBE"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "targets : ${TARGETS_JSON}"
echo "dataset : ${DATASET_DIR}"
echo "frames  : ${N_FRAMES}  bs=${BATCH_SIZE}  sigmas=${GAUSS_SIGMAS}"
echo "out     : ${OUTPUT_DIR}"
nvidia-smi || true

if [ ! -e "${DATASET_DIR}" ]; then
  echo "skillvla dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/input_probe.py" \
    --targets="${TARGETS_JSON}" \
    --dataset_dir="${DATASET_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --n_frames="${N_FRAMES}" \
    --batch_size="${BATCH_SIZE}" \
    --gauss_sigmas="${GAUSS_SIGMAS}" \
    --seed="${SEED}"

# Render the readable comparison table.
"${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/report.py" --output_dir="${OUTPUT_DIR}"

echo "DONE → ${OUTPUT_DIR}/  (input_probe.json + summary.md)"
