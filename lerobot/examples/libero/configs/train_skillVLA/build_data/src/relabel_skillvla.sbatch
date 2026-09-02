#!/usr/bin/env bash
# Relabel one completed SkillVLA dataset with a frozen skill predictor.
#SBATCH --job-name=skillVLA_relabel
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --relabel --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --relabel --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA RELABEL"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Source     : ${RELABEL_SOURCE_RUN_DIR}"
echo "Predictor  : ${RELABEL_PREDICTOR_MODEL}/${RELABEL_PREDICTOR_CHECKPOINT}"
echo "Output     : ${RELABEL_OUTPUT_RUN_DIR}"
echo "Code space : ${RELABEL_CODE_SPACE_ID}"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u "${SRC_DIR}/relabel_skillvla.py" \
    --source-run-dir "${RELABEL_SOURCE_RUN_DIR}" \
    --output-run-dir "${RELABEL_OUTPUT_RUN_DIR}" \
    --predictor-path "${RELABEL_PREDICTOR_PATH}" \
    --predictor-model "${RELABEL_PREDICTOR_MODEL}" \
    --predictor-checkpoint "${RELABEL_PREDICTOR_CHECKPOINT}" \
    --tokenizer-path "${RELABEL_TOKENIZER_PATH}" \
    --code-space-id "${RELABEL_CODE_SPACE_ID}" \
    --batch-size "${RELABEL_BATCH_SIZE}"

