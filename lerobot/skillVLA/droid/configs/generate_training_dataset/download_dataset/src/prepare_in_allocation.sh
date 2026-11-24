#!/usr/bin/env bash
# Called only by the original allocation's shared worker, not an extra srun.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
EXPECTED_JOB="${1:?Missing original job ID}"
FROZEN_CONFIG="${2:?Missing frozen configuration}"
CPU_BUDGET="${3:?Missing CPU budget}"
shift 3
[[ "${SLURM_JOB_ID:-}" == "${EXPECTED_JOB}" ]] || {
  echo "Refusing to convert outside the requested original Slurm allocation." >&2; exit 1;
}
[[ "${SLURMD_NODENAME:-}" == "$(hostname -s)" ]] || {
  echo "Conversion is not on the allocated node." >&2; exit 1;
}
# CPU-only work; do not enroll this process in a shared GPU memory guard.
export CUDA_VISIBLE_DEVICES="" SBD_GPU_MEMORY_GUARD=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
# Restrict only this conversion child. Other jobs and their affinities are untouched.
exec "${PYTHON}" - "${SCRIPT_DIR}/prepare_dataset.py" "${FROZEN_CONFIG}" "${CPU_BUDGET}" "$@" <<'PY'
import os, sys
script, config, budget, *extra = sys.argv[1:]
budget = int(budget)
allowed = sorted(os.sched_getaffinity(0))
if budget < 1 or budget > len(allowed):
    raise SystemExit(f'Conversion needs {budget} CPUs but this allocation shell can use {len(allowed)}; refusing to oversubscribe.')
os.sched_setaffinity(0, allowed[:budget])
print(f'DROID CPU conversion: job={os.environ["SLURM_JOB_ID"]}, CPUs={budget}, GPU disabled', flush=True)
os.execv(sys.executable, [sys.executable, script, '--config', config, *extra])
PY
