#!/usr/bin/env bash
# Stage-1 STATE-influence probe (offline, no simulator): perturb the state and measure the action
# chunk change vs skill-swap / image-zero references. See src/state_probe.py.
#SBATCH --job-name=state_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${STATE_PROBE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # state_probe
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STATE_PROBE_CONFIG:-${SUBMIT_DIR}/state_probe_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/state_probe_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

echo "START STATE PROBE (pi05 vs cond)"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "cond    : ${COND_POLICY_PATH}"
echo "pi05    : ${PI05_POLICY_PATH:-(none → cond-only)}"
echo "dataset : ${DATASET_DIR}"
echo "frames  : ${N_FRAMES}  bs=${BATCH_SIZE}  sigmas=${GAUSS_SIGMAS}"
echo "out     : ${OUTPUT_DIR}"
nvidia-smi || true

if [ ! -d "${COND_POLICY_PATH}" ]; then
  echo "cond (stage-1) checkpoint not found: ${COND_POLICY_PATH}" >&2
  exit 1
fi
if [ -n "${PI05_POLICY_PATH}" ] && [ ! -d "${PI05_POLICY_PATH}" ]; then
  echo "pi05 checkpoint not found: ${PI05_POLICY_PATH}" >&2
  exit 1
fi
if [ ! -e "${DATASET_DIR}" ]; then
  echo "skillvla dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/state_probe.py" \
    --cond_policy_path="${COND_POLICY_PATH}" \
    --pi05_policy_path="${PI05_POLICY_PATH}" \
    --dataset_dir="${DATASET_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --n_frames="${N_FRAMES}" \
    --batch_size="${BATCH_SIZE}" \
    --gauss_sigmas="${GAUSS_SIGMAS}" \
    --seed="${SEED}"

# Render the readable comparison table.
"${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/report.py" --output_dir="${OUTPUT_DIR}"

echo "DONE → ${OUTPUT_DIR}/  (state_probe.json + summary.md)"
