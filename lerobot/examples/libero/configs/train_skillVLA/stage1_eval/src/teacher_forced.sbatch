#!/usr/bin/env bash
# SkillVLA Stage-1 OFFLINE teacher-forced eval (no simulator).
# GT dataset frames + GT skill code → chunk accuracy + skill-swap sensitivity.
#SBATCH --job-name=skillExpert_S1_TF
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${STAGE1_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # stage1_eval
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SUBMIT_DIR}/stage1_eval_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

OUTPUT_DIR="${EVAL_OUT_DIR}/teacher_forced"

echo "START STAGE-1 TEACHER-FORCED EVAL (skill_expert, offline)"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "policy   : ${POLICY_PATH}"
echo "dataset  : ${SKILL_LABEL_DATASET_DIR}"
echo "output   : ${OUTPUT_DIR}"
nvidia-smi || true

PYTHONPATH="${LEROBOT_ROOT}/src" \
  python -u "${SRC_DIR}/teacher_forced.py" \
    --policy_path "${POLICY_PATH}" \
    --dataset_dir "${SKILL_LABEL_DATASET_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --n_frames    "${TF_N_FRAMES:-512}" \
    --batch_size  "${TF_BATCH_SIZE:-32}" \
    --n_swap      "${TF_N_SWAP:-4}" \
    --seed        1000

echo "DONE → ${OUTPUT_DIR}/teacher_forced.json"
