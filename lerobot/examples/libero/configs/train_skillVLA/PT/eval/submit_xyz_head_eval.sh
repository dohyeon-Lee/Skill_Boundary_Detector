#!/usr/bin/env bash
# Submit simulator-free diffaux3/4 XYZ-head evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
CONFIG_LIB="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${XYZ_HEAD_EVAL_CONFIG:-${SCRIPT_DIR}/xyz_head_eval_config.yaml}"
MODE=submit
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG_PATH="${2:?--config needs a YAML path}"; shift 2 ;;
    --dry-run) MODE=check; shift ;;
    --local) MODE=local; shift ;;
    -h|--help)
      echo "Usage: $0 [--config FILE] [--dry-run | --local]"
      echo "Default: one GPU array worker per model, capped by eval_num_gpus."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -x "${PYTHON}" ] || { echo "Missing ${PYTHON}; run setup_env.sh first." >&2; exit 1; }
if [ "${MODE}" = local ]; then
  exec "${PYTHON}" "${SCRIPT_DIR}/src/xyz_head_eval.py" --config "${CONFIG_PATH}"
fi
if [ "${MODE}" = check ]; then
  exec "${PYTHON}" "${SCRIPT_DIR}/src/xyz_head_eval_config.py" --config "${CONFIG_PATH}"
fi

source "${CONFIG_LIB}/src/submit_job.sh"
source "${CONFIG_LIB}/src/snapshot_config.sh"
"${PYTHON}" "${SCRIPT_DIR}/src/xyz_head_eval_config.py" --config "${CONFIG_PATH}"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
eval "$("${PYTHON}" "${SCRIPT_DIR}/src/xyz_head_eval_config.py" --config "${CONFIG_PATH}" --shell)"

mkdir -p "${SCRIPT_DIR}/logs"
SBATCH_ARGS=(
  --job-name=S1xyz
  --cpus-per-task="${XYZ_HEAD_CPUS}"
  --mem="${XYZ_HEAD_MEM}"
  --time="${XYZ_HEAD_TIME}"
)
[ -z "${XYZ_HEAD_PARTITION}" ] || SBATCH_ARGS+=(--partition="${XYZ_HEAD_PARTITION}")
[ -z "${XYZ_HEAD_QOS}" ] || SBATCH_ARGS+=(--qos="${XYZ_HEAD_QOS}")
if [ "${XYZ_HEAD_DEVICE}" = cuda ]; then
  SBATCH_ARGS+=(--gres=gpu:1)
  [ -z "${XYZ_HEAD_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${XYZ_HEAD_NODELIST}")
  [ -z "${XYZ_HEAD_EXCLUDE}" ] || SBATCH_ARGS+=(--exclude="${XYZ_HEAD_EXCLUDE}")
fi

cd "${SCRIPT_DIR}"
export XYZ_HEAD_EVAL_CONFIG="${CONFIG_PATH}"
export XYZ_HEAD_PROJECT_ROOT="${PROJECT_ROOT}"
export XYZ_HEAD_EVAL_SRC="${SCRIPT_DIR}/src"
export XYZ_HEAD_DEVICE XYZ_HEAD_WORKER_COUNT
echo "XYZ-head offline evaluation: ${XYZ_HEAD_WORKER_COUNT} worker(s)."
echo "  output: ${XYZ_HEAD_OUTPUT_DIR}"
if [ "${XYZ_HEAD_WORKER_COUNT}" -le 1 ]; then
  export XYZ_HEAD_WORKER_INDEX=0
  submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/xyz_head_eval.sbatch"
elif [ "$(submit_job_scheduler)" = slurm ]; then
  submit_job --array="0-$((XYZ_HEAD_WORKER_COUNT - 1))%${XYZ_HEAD_WORKER_COUNT}" \
    --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
    "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/xyz_head_eval.sbatch"
else
  for ((worker=0; worker<XYZ_HEAD_WORKER_COUNT; worker++)); do
    export XYZ_HEAD_WORKER_INDEX="${worker}"
    submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/xyz_head_eval.sbatch"
  done
fi
