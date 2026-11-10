#!/usr/bin/env bash
# Complete the build inside one GPU allocation; segmentation workers share it.
#SBATCH --job-name=skillVLA_build_1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
else
  CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
  [ -z "${SOURCE_DATA:-}" ] || CONFIG_ARGS+=(--dataset "${SOURCE_DATA}")
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" "${CONFIG_ARGS[@]}")"
  eval "${RESOLVED_SETTINGS}"
  mkdir -p "${SUBMIT_DIR}/logs"
  export SKILLVLA_ENV_SNAPSHOT="${SUBMIT_DIR}/logs/skillvla_single_gpu_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
  printf '%s\n' "${RESOLVED_SETTINGS}" > "${SKILLVLA_ENV_SNAPSHOT}"
fi

source "${SRC_DIR}/skillset_args.sh"
skillvla_check_contract preflight
source "${PROJECT_ROOT}/.venv/bin/activate"
# Child GPU guards must never requeue/cancel a shared allocation on failure.
export SBD_SRUN_SESSION=1
if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" != 1 ]; then
  "${PROJECT_ROOT}/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Single GPU build requires exactly one visible CUDA device"'
fi

exec python -u "${SRC_DIR}/run_single_gpu_build.py"
