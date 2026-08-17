#!/usr/bin/env bash
# Evaluate one FSQ-original checkpoint (env: FSQ_RUN_NAME, FSQ_EPOCH_TAG).
# Outputs: skill_eval/outputs/fsq_original_eval/{run}/{epoch_tag}/metrics.json + html
#SBATCH --job-name=FSQ_orig_eval
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH=${FSQ_ORIG_EVAL_CONFIG:-${SUBMIT_DIR}/fsq_original_eval_config.yaml}

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_original_eval_config.py" --config "${CONFIG_PATH}" --shell)"
eval "${RESOLVED_SETTINGS}"

: "${FSQ_RUN_NAME:?FSQ_RUN_NAME is required}"
: "${FSQ_EPOCH_TAG:?FSQ_EPOCH_TAG is required}"

RUN_DIR="${FSQ_OUTPUTS_ROOT}/${FSQ_RUN_NAME}"
MODEL_PATH="${RUN_DIR}/FSQ_${FSQ_EPOCH_TAG}.pt"
SKILLS_DIR="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_original_eval_config.py" --config "${CONFIG_PATH}" --resolve-skills "${RUN_DIR}")"
OUTPUT_DIR="${SUBMIT_DIR}/outputs/fsq_original_eval/${FSQ_RUN_NAME}/${FSQ_EPOCH_TAG}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START FSQ-ORIGINAL EVAL"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "Model   : ${MODEL_PATH}"
echo "Skills  : ${SKILLS_DIR}"
echo "Output  : ${OUTPUT_DIR}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/fsq_original_eval.py \
    --model_path      "${MODEL_PATH}" \
    --skills_dir      "${SKILLS_DIR}" \
    --output_dir      "${OUTPUT_DIR}" \
    --term_threshold  "${FSQ_ORIG_EVAL_TERM_THRESHOLD}" \
    --val_split       "${FSQ_ORIG_EVAL_VAL_SPLIT}"

echo "DONE ${FSQ_RUN_NAME} ${FSQ_EPOCH_TAG}"
