#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${ROOT}/setup_env.sh" ]]; do ROOT="$(dirname "${ROOT}")"; done
if [[ "${SBD_GPU_MEMORY_GUARD:-false}" == @(1|true|True) ]] && [[ "${SBD_GPU_GUARD_CHILD:-0}" != 1 ]]; then
  exec "${ROOT}/.venv/bin/python" "${ROOT}/lerobot/skillVLA/libero/configs/src/guard_eval.py" "$0" "$@"
fi
export PYTHONPATH="${ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset LD_LIBRARY_PATH
eval "$("${ROOT}/.venv/bin/python" -m lerobot.utils.eval_gpu_memory --workers 1 --shell)"
exec "${ROOT}/.venv/bin/python" "$(dirname "${BASH_SOURCE[0]}")/skill_start_probe.py" --settings "${PROBE_SETTINGS}"
