#!/usr/bin/env bash
#SBATCH --job-name=ABC_skillVLA_encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$(${BOOTSTRAP_PYTHON} "${SUBMIT_DIR}/src/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/skillVLA/abc/configs/src/runtime_env.sh"
source "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

[ -f "${FSQ_MODEL_PATH}" ] || { echo "FSQ checkpoint missing: ${FSQ_MODEL_PATH}" >&2; exit 1; }
mkdir -p "${SKILLVLA_RUN_DIR}"
echo "ABC SkillVLA FSQ encode"
echo "  skillset : ${SKILLSET_DIR}/skills"
echo "  FSQ      : ${FSQ_MODEL_PATH}"
echo "  output   : ${SKILL_LATENTS_PATH}"
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u skillVLA/libero/configs/train_skillVLA/build_data/src/encode_FSQ_skills.py \
    --model_path "${FSQ_MODEL_PATH}" \
    --skills_dir "${SKILLSET_DIR}/skills" \
    --output_path "${SKILL_LATENTS_PATH}" \
    --device cuda \
    --no-snap_to_supported
echo "DONE -> ${SKILL_LATENTS_PATH}"
