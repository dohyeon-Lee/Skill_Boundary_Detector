#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
CONFIG="${ABC_DP_EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"
MODE=submit
REPORT_ONLY=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG="${2:?--config requires a YAML path}"; shift 2 ;;
    --dry-run) MODE=plan; shift ;;
    --local) MODE=local; shift ;;
    --report-only) REPORT_ONLY=true; MODE=local; shift ;;
    *) echo "Usage: $0 [--config FILE] [--dry-run|--local|--report-only]" >&2; exit 2 ;;
  esac
done
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
# Resolve and execute the same frozen YAML, even if the live file is edited
# while checkpoint/cache validation is still running.
if [ "${MODE}" != plan ]; then
  source "${SCRIPT_DIR}/../../../src/snapshot_config.sh"
  CONFIG="$(snapshot_config "${CONFIG}")"
fi
if ! RESOLVED="$("${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --shell)"; then
  echo "DP evaluation config failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"
export EVAL_OUT_DIR="${DP_ABLATION_OUTPUT_DIR}"
echo "Episodes: ${DP_ABLATION_EPISODE_COUNT:?standalone: true is required}"
echo "DP: history=${DP_ABLATION_N_OBS_STEPS}, max prediction=${DP_ABLATION_PREDICTION_HORIZON}; SBD uses ${DP_ABLATION_SBD_PREDICTION_HORIZON} step(s)"
echo "Partial build: ${DP_ABLATION_ARTIFACT_DIR}"
echo "Report: ${DP_ABLATION_OUTPUT_DIR}/index.html"
echo "Cache: ${DP_ABLATION_CACHE_STATUS}"
[ "${MODE}" = plan ] && exit 0
if [ "${DP_ABLATION_CACHE_READY}" = true ]; then
  REPORT_ONLY=true
  MODE=local
fi
if [ "${MODE}" = local ]; then
  EXTRA=()
  [ "${REPORT_ONLY}" = true ] && EXTRA+=(--report-only)
  [ "${REPORT_ONLY}" = true ] && export CUDA_VISIBLE_DEVICES=""
  if [ "${REPORT_ONLY}" != true ] && [ "${DP_ABLATION_DEVICE}" = cuda ]; then
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "Run --local inside a GPU allocation, or submit without --local"'
  fi
  source "${SCRIPT_DIR}/../../../src/runtime_env.sh"
  export OMP_NUM_THREADS="${DP_ABLATION_CPUS}"
  export MKL_NUM_THREADS="${DP_ABLATION_CPUS}"
  export OPENBLAS_NUM_THREADS="${DP_ABLATION_CPUS}"
  exec "${PYTHON}" -u "${SCRIPT_DIR}/src/run_eval.py" --config "${CONFIG}" "${EXTRA[@]}"
fi
source "${SCRIPT_DIR}/../../../src/submit_job.sh"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
SBATCH_ARGS=(--job-name=ABC_DP_SBD_eval --partition="${DP_ABLATION_PARTITION}" --qos="${DP_ABLATION_QOS}"
  --cpus-per-task="${DP_ABLATION_CPUS}" --mem="${DP_ABLATION_MEM}" --time="${DP_ABLATION_TIME}"
  --export="ALL,ABC_DP_EVAL_DIR=${SCRIPT_DIR},ABC_DP_EVAL_CONFIG=${CONFIG}")
[ "${DP_ABLATION_DEVICE}" = cuda ] && SBATCH_ARGS+=(--gres="${DP_ABLATION_GRES}")
[ -n "${DP_ABLATION_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${DP_ABLATION_NODELIST}")
[ -n "${DP_ABLATION_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${DP_ABLATION_EXCLUDE_NODES}")
submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/run_eval.sbatch"
