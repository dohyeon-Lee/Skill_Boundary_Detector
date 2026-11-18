#!/usr/bin/env bash
#SBATCH --job-name=abc_sim_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

EVAL_DIR="${ABC_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SRC_DIR="${ABC_EVAL_SRC_DIR:-${EVAL_DIR}/src}"
CONFIG_PATH="${ABC_EVAL_CONFIG:-${EVAL_DIR}/abc_sim_eval_config.yaml}"
PROJECT_ROOT="${ABC_EVAL_PROJECT_ROOT:-$(cd "${EVAL_DIR}/../../../../../../.." && pwd)}"
PYTHON="${PROJECT_ROOT}/.venv_abc_sim/bin/python"

if [ ! -x "${PYTHON}" ]; then
  echo "ABC Sim environment is missing: ${PYTHON}; run bash setup_env.sh --abc-sim-only" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
first_visible_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
if [[ "${first_visible_gpu}" =~ ^[0-9]+$ ]]; then
  export MUJOCO_EGL_DEVICE_ID="${first_visible_gpu}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${EVAL_DIR}/logs"

echo "START ABC SIM EVAL"
echo "node    : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "cuda    : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "config  : ${CONFIG_PATH}"
echo "output  : ${ABC_EVAL_OUTPUT_DIR:-resolved by config}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

exec "${PYTHON}" "${SRC_DIR}/run_eval.py" --config "${CONFIG_PATH}"
