#!/usr/bin/env bash
#SBATCH --job-name=ABC_SkillVLA_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

EVAL_DIR="${ABC_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${ABC_EVAL_PROJECT_ROOT:-$(cd "${EVAL_DIR}/../../../../../../.." && pwd)}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MUJOCO_GL=egl
# Use the training env for SkillVLA's exact Transformers/processor versions.
# ABC source and simulator dependencies are available without reinstalling it.
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/third_party/abc${PYTHONPATH:+:${PYTHONPATH}}"
first_visible_gpu="${CUDA_VISIBLE_DEVICES:-}"
first_visible_gpu="${first_visible_gpu%%,*}"
if [[ "${first_visible_gpu}" =~ ^[0-9]+$ ]]; then
  export MUJOCO_EGL_DEVICE_ID="${first_visible_gpu}"
fi
cd "${PROJECT_ROOT}"
echo "START ABC SkillVLA evaluation: job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-${HOSTNAME}}"
exec "${PROJECT_ROOT}/.venv/bin/python" "${EVAL_DIR}/src/evaluate.py" \
  --config "${ABC_EVAL_CONFIG:-${EVAL_DIR}/pt_eval_config.yaml}" "$@"
