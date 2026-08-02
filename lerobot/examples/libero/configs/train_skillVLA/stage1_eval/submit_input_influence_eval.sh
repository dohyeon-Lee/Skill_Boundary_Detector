#!/usr/bin/env bash
# Submit one Stage-1 input-influence worker per selected model, then aggregate.
set -euo pipefail

# Slurm copies this script into /var/spool before executing it. Preserve the
# submit-side source directory through the exported environment instead of
# resolving BASH_SOURCE again inside an array worker.
SCRIPT_DIR="${STAGE1_INPUT_INFLUENCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SRC_DIR="${SCRIPT_DIR}/src"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
LEROBOT_ROOT="${PROJECT_ROOT}/lerobot"
BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
MODE="${1:-submit}"
CONFIG_PATH="${2:-${STAGE1_INPUT_INFLUENCE_CONFIG:-${SCRIPT_DIR}/input_influence_eval_config.yaml}}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
fi

resolve_config() {
  local exports
  exports="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" \
      --config "${CONFIG_PATH}" --shell
  )"
  eval "${exports}"
}

activate_runtime() {
  cd "${LEROBOT_ROOT}"
  source "${PROJECT_ROOT}/.venv/bin/activate"
  source "${LEROBOT_ROOT}/examples/libero/configs/runtime_env.sh"
  unset LD_LIBRARY_PATH
}

if [ "${MODE}" = "worker" ]; then
  resolve_config
  activate_runtime
  MODEL_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  echo "START STAGE-1 INPUT INFLUENCE"
  echo "node   : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
  echo "job    : ${SLURM_JOB_ID:-none} array=${MODEL_INDEX}"
  echo "cuda   : ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "config : ${CONFIG_PATH}"
  echo "output : ${EVAL_OUT_DIR}/input_influence"
  nvidia-smi || true
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="${LEROBOT_ROOT}/src:${SRC_DIR}" \
    "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/input_influence_eval.py" \
      --config "${CONFIG_PATH}" --model-index "${MODEL_INDEX}"
  exit 0
fi

if [ "${MODE}" = "aggregate" ]; then
  resolve_config
  activate_runtime
  PYTHONPATH="${LEROBOT_ROOT}/src:${SRC_DIR}" \
    "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/input_influence_eval.py" \
      --config "${CONFIG_PATH}" --aggregate
  echo "SUMMARY -> ${EVAL_OUT_DIR}/input_influence/summary.html"
  exit 0
fi

if [ "${MODE}" != "submit" ]; then
  echo "Usage: $0 [submit|worker|aggregate] [config.yaml]" >&2
  exit 2
fi

SNAPSHOT_LIB="${SCRIPT_DIR}/../../snapshot_config.sh"
source "${SNAPSHOT_LIB}"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
resolve_config

if [ "${MODEL_COUNT}" -le 0 ]; then
  echo "No models selected." >&2
  exit 1
fi
PARALLELISM="${EVAL_NUM_GPUS}"
if [ "${PARALLELISM}" -gt "${MODEL_COUNT}" ]; then
  PARALLELISM="${MODEL_COUNT}"
fi
if [ "${PARALLELISM}" -lt 1 ]; then
  PARALLELISM=1
fi

SBATCH_COMMON=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_COMMON+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_COMMON+=(--exclude="${EVAL_EXCLUDE_NODES}")
mkdir -p "${SCRIPT_DIR}/logs"
export STAGE1_INPUT_INFLUENCE_DIR="${SCRIPT_DIR}"

ARRAY_SUBMISSION="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --export="ALL,STAGE1_INPUT_INFLUENCE_DIR=${SCRIPT_DIR}" \
    --job-name=S1input \
    --array="0-$((MODEL_COUNT - 1))%${PARALLELISM}" \
    --gres="${EVAL_GRES}" \
    --cpus-per-task="${EVAL_CPUS_PER_TASK}" \
    --mem="${EVAL_MEM}" \
    --time="${EVAL_TIME}" \
    --output="${SCRIPT_DIR}/logs/%x_%A_%a.out" \
    --error="${SCRIPT_DIR}/logs/%x_%A_%a.err" \
    "$0" worker "${CONFIG_PATH}"
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
SUMMARY_SUBMISSION="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --export="ALL,STAGE1_INPUT_INFLUENCE_DIR=${SCRIPT_DIR}" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --job-name=S1input_summary \
    --gres="${EVAL_GRES}" \
    --cpus-per-task=2 --mem=8G --time=30:00 \
    --output="${SCRIPT_DIR}/logs/%x_%j.out" \
    --error="${SCRIPT_DIR}/logs/%x_%j.err" \
    "$0" aggregate "${CONFIG_PATH}"
)"

echo "Submitted Stage-1 input-influence evaluation"
echo "  models     : ${MODEL_COUNT}"
echo "  GPUs       : ${PARALLELISM} concurrent"
echo "  array job  : ${ARRAY_SUBMISSION}"
echo "  summary job: ${SUMMARY_SUBMISSION}"
echo "  output     : ${EVAL_OUT_DIR}/input_influence"
