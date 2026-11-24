#!/usr/bin/env bash
#SBATCH --job-name=MimicGen_pi05_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
CODE_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PYTHON="${PYTHON_BIN:?submit with submit_eval.sh}"
export PYTHONPATH="${CODE_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${SBD_GPU_MEMORY_GUARD:-false}" == @(1|true|True) ]] && [ "${SBD_GPU_GUARD_CHILD:-0}" != 1 ]; then
  exec "${PYTHON}" "${CODE_ROOT}/lerobot/skillVLA/libero/configs/src/guard_eval.py" "$0" "$@"
fi
eval "$("${PYTHON}" "${HERE}/src/config.py" --config "${PI05_EVAL_CONFIG}" --shell)"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${CODE_ROOT}/lerobot/skillVLA/libero/configs/src/runtime_env.sh"
export OMP_NUM_THREADS="${EVAL_CPUS_PER_TASK}" MKL_NUM_THREADS="${EVAL_CPUS_PER_TASK}"
export MIMICGEN_SOURCE_ROOT EPISODE_EXACT_REPEAT="${REPEAT_EPISODES}"
exec "${PYTHON}" "${HERE}/src/run_eval.py"
