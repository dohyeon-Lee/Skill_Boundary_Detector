#!/usr/bin/env bash
# Offline Predictor report, independent of the closed-loop FT evaluation engine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
CONFIG_LIB="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${FT_PRED_EVAL_CONFIG:-${SCRIPT_DIR}/ft_pred_eval_config.yaml}"
MODE=submit
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG_PATH="${2:?--config needs a YAML path}"; shift 2 ;;
    --dry-run) MODE=check; shift ;;
    --local) MODE=local; shift ;;
    -h|--help)
      echo "Usage: $0 [--config FILE] [--dry-run | --local]"
      echo "Default: split checkpoints over up to eval_num_gpus one-GPU jobs."
      echo "--local: run sequentially on one GPU/CPU (compute node only)."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ ! -x "${PYTHON}" ]; then
  echo "Missing ${PYTHON}; run setup_env.sh first." >&2
  exit 1
fi
if [ "${MODE}" = local ]; then
  exec "${PYTHON}" "${SCRIPT_DIR}/src/ft_pred_eval.py" --config "${CONFIG_PATH}"
fi
if [ "${MODE}" = check ]; then
  exec "${PYTHON}" "${SCRIPT_DIR}/src/ft_pred_eval_config.py" --config "${CONFIG_PATH}"
fi
source "${CONFIG_LIB}/src/submit_job.sh"
source "${CONFIG_LIB}/src/snapshot_config.sh"
# Preflight before creating a snapshot or submitting anything. In particular,
# a map without the completed SkillVLA build is not sufficient.
"${PYTHON}" "${SCRIPT_DIR}/src/ft_pred_eval_config.py" --config "${CONFIG_PATH}"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
eval "$("${PYTHON}" "${SCRIPT_DIR}/src/ft_pred_eval_config.py" --config "${CONFIG_PATH}" --shell)"
mkdir -p "${SCRIPT_DIR}/logs"
SBATCH_ARGS=(--job-name=FTpred --cpus-per-task="${FT_PRED_CPUS}" --mem="${FT_PRED_MEM}" --time="${FT_PRED_TIME}")
[ -z "${FT_PRED_PARTITION}" ] || SBATCH_ARGS+=(--partition="${FT_PRED_PARTITION}")
[ -z "${FT_PRED_QOS}" ] || SBATCH_ARGS+=(--qos="${FT_PRED_QOS}")
if [ "${FT_PRED_DEVICE}" = cuda ]; then
  SBATCH_ARGS+=(--gres=gpu:1)
  [ -z "${FT_PRED_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${FT_PRED_NODELIST}")
  [ -z "${FT_PRED_EXCLUDE}" ] || SBATCH_ARGS+=(--exclude="${FT_PRED_EXCLUDE}")
fi
cd "${SCRIPT_DIR}"
export FT_PRED_EVAL_CONFIG="${CONFIG_PATH}"
export FT_PRED_EVAL_PROJECT_ROOT="${PROJECT_ROOT}"
export FT_PRED_EVAL_SRC="${SCRIPT_DIR}/src"
export FT_PRED_DEVICE
export FT_PRED_WORKER_COUNT
echo "FT Predictor evaluation: ${FT_PRED_WORKER_COUNT} worker(s), one checkpoint group per worker."
echo "  output: ${FT_PRED_OUTPUT_DIR}"
if [ "${FT_PRED_WORKER_COUNT}" -le 1 ]; then
  export FT_PRED_WORKER_INDEX=0
  submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/pred_eval.sbatch"
elif [ "$(submit_job_scheduler)" = slurm ]; then
  # Each array element can start as soon as one GPU is available; do not wait
  # for all GPUs to fit on one node/in one allocation.
  submit_job --array="0-$((FT_PRED_WORKER_COUNT - 1))%${FT_PRED_WORKER_COUNT}" \
    --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
    "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/pred_eval.sbatch"
else
  # The local scheduler does not support Slurm arrays; reserve one GPU per job.
  # Like other local launchers, it fails if insufficient free GPUs remain.
  for ((worker=0; worker<FT_PRED_WORKER_COUNT; worker++)); do
    export FT_PRED_WORKER_INDEX="${worker}"
    submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/pred_eval.sbatch"
  done
fi
