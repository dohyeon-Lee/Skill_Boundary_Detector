#!/usr/bin/env bash
#SBATCH --job-name=LG_init
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
set -euo pipefail

HERE="${ORACLE_MATCHING_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${HERE}" ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
mkdir -p "${HERE}/../logs" "${PROJECT_ROOT}/.cache/numba" "${PROJECT_ROOT}/.cache/matplotlib"

source "${PROJECT_ROOT}/.venv/bin/activate"
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/.cache/numba"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/tools/lerobot-libero/.libero_config"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/tools/lerobot-libero${PYTHONPATH:+:${PYTHONPATH}}"

exec "${HERE}/run.sh" "$@"
