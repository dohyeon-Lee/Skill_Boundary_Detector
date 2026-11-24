#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
SHARED="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src"
export SBD_GLOBAL_CONFIG="${SCRIPT_DIR}/../../../global_config.yaml"
source "${SHARED}/submit_args.sh"
CONFIG="${DROID_DP_EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"
MODE=submit
REPORT_ONLY=false
while (( $# )); do
  case "$1" in
    --config) CONFIG="${2:?--config requires a YAML path}"; shift 2 ;;
    --dry-run) MODE=plan; shift ;;
    --local) MODE=local; shift ;;
    --report-only) REPORT_ONLY=true; MODE=local; shift ;;
    *) echo "Usage: $0 [--JOB_ID] [--config FILE] [--dry-run|--local|--report-only]" >&2; exit 2 ;;
  esac
done
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${MODE}" != plan ]]; then
  source "${SHARED}/snapshot_config.sh"
  CONFIG="$(snapshot_config "${CONFIG}")"
  export SBD_GLOBAL_CONFIG="$(dirname "${CONFIG}")/global_config.yaml"
fi
if ! RESOLVED="$("${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --shell)"; then
  echo "DROID DP evaluation prerequisites/config failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"
export EVAL_OUT_DIR="${DP_ABLATION_OUTPUT_DIR}"
echo "Episodes: ${DP_ABLATION_EPISODE_COUNT}"
echo "DP: history=${DP_ABLATION_N_OBS_STEPS}, prediction=${DP_ABLATION_PREDICTION_HORIZON}; SBD=${DP_ABLATION_SBD_PREDICTION_HORIZON}"
echo "Partial build: ${DP_ABLATION_ARTIFACT_DIR}"
echo "Report: ${DP_ABLATION_OUTPUT_DIR}/index.html"
echo "Cache: ${DP_ABLATION_CACHE_STATUS}"
[[ "${MODE}" != plan ]] || exit 0
if [[ "${DP_ABLATION_CACHE_READY}" == true ]]; then
  REPORT_ONLY=true
  MODE=local
fi
if [[ "${MODE}" == local ]]; then
  EXTRA=()
  [[ "${REPORT_ONLY}" != true ]] || { EXTRA+=(--report-only); export CUDA_VISIBLE_DEVICES=""; }
  if [[ "${REPORT_ONLY}" != true && "${DP_ABLATION_DEVICE}" == cuda ]]; then
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "Run --local inside a GPU allocation, or submit with --JOB_ID"'
  fi
  source "${SHARED}/runtime_env.sh"
  export OMP_NUM_THREADS="${DP_ABLATION_CPUS}" MKL_NUM_THREADS="${DP_ABLATION_CPUS}" OPENBLAS_NUM_THREADS="${DP_ABLATION_CPUS}"
  exec "${PYTHON}" -u "${SCRIPT_DIR}/src/run_eval.py" --config "${CONFIG}" "${EXTRA[@]}"
fi
source "${SHARED}/submit_job.sh"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
OPTIONS=(--job-name=DROID_DP_SBD_eval --partition="${DP_ABLATION_PARTITION}" --qos="${DP_ABLATION_QOS}"
  --cpus-per-task="${DP_ABLATION_CPUS}" --mem="${DP_ABLATION_MEM}" --time="${DP_ABLATION_TIME}"
  --export="ALL,DROID_DP_EVAL_DIR=${SCRIPT_DIR},DROID_DP_EVAL_CONFIG=${CONFIG}")
[[ "${DP_ABLATION_DEVICE}" != cuda ]] || OPTIONS+=(--gres="${DP_ABLATION_GRES}")
[[ -z "${DP_ABLATION_NODELIST}" ]] || OPTIONS+=(--nodelist="${DP_ABLATION_NODELIST}")
[[ -z "${DP_ABLATION_EXCLUDE_NODES}" ]] || OPTIONS+=(--exclude="${DP_ABLATION_EXCLUDE_NODES}")
submit_job "${OPTIONS[@]}" "${SCRIPT_DIR}/src/run_eval.sbatch"
