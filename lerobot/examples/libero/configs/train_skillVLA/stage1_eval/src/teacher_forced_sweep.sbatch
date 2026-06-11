#!/usr/bin/env bash
# Teacher-forced eval swept over MULTIPLE checkpoints of one Stage-1 run (one GPU, sequential).
# Produces outputs/{model_dir}_{ckpt}_adv-{mode}/teacher_forced/teacher_forced.json per checkpoint
# (already-done checkpoints are skipped → resubmit anytime as training progresses), then prints
# the win_rate/swap_delta trend table (src/tf_trend.py).
#SBATCH --job-name=skillExpert_S1_TFsweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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
unset LD_LIBRARY_PATH

# model root from the resolved policy path: .../skillVLA_stage1/{model_dir}/checkpoints/{ckpt}/pretrained_model
MODEL_ROOT="$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")"
MODEL_DIR="$(basename "${MODEL_ROOT}")"

# TF_CKPTS="005000 025000 last" overrides; default = every numeric checkpoint, oldest first.
if [ -n "${TF_CKPTS:-}" ]; then
  CKPTS=(${TF_CKPTS})
else
  CKPTS=($(ls "${MODEL_ROOT}/checkpoints" | grep -E '^[0-9]+$' | sort -n))
fi

echo "START STAGE-1 TEACHER-FORCED SWEEP"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "model    : ${MODEL_DIR}"
echo "ckpts    : ${CKPTS[*]}"
echo "dataset  : ${SKILL_LABEL_DATASET_DIR}"
nvidia-smi || true

for CKPT in "${CKPTS[@]}"; do
  POLICY="${MODEL_ROOT}/checkpoints/${CKPT}/pretrained_model"
  OUT="${SUBMIT_DIR}/outputs/${MODEL_DIR}_${CKPT}_adv-${SKILL_ADVANCE_MODE}/teacher_forced"
  if [ -f "${OUT}/teacher_forced.json" ]; then
    echo "== ${CKPT}: already done, skipping =="
    continue
  fi
  if [ ! -d "${POLICY}" ]; then
    echo "== ${CKPT}: no pretrained_model dir, skipping =="
    continue
  fi
  echo "== ${CKPT}: running =="
  PYTHONPATH="${LEROBOT_ROOT}/src" \
    python -u "${SRC_DIR}/teacher_forced.py" \
      --policy_path "${POLICY}" \
      --dataset_dir "${SKILL_LABEL_DATASET_DIR}" \
      --output_dir  "${OUT}" \
      --n_frames    "${TF_N_FRAMES:-512}" \
      --batch_size  "${TF_BATCH_SIZE:-32}" \
      --n_swap      "${TF_N_SWAP:-4}" \
      --seed        1000
done

PYTHONPATH="${LEROBOT_ROOT}/src" \
  python "${SRC_DIR}/tf_trend.py" --model_dir "${MODEL_DIR}" --outputs_dir "${SUBMIT_DIR}/outputs"
echo "DONE"
