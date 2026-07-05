#!/usr/bin/env bash
# Submit side-by-side compare eval using Slurm settings from ./cycle_eval_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_PI05_CYCLE_EVAL_CONFIG:-${SCRIPT_DIR}/cycle_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_cycle_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem=48G
  --time="${EVAL_TIME}"
)
[ -n "${EVAL_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -n "${EVAL_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

# 샤드>1 → SLURM array: task들을 GPU N개에 round-robin 분배, 끝나면 merge 잡 자동 제출.
# 값은 yaml compare_n_shards 또는 env N_SHARDS (env 우선, 리졸버가 처리).
N_SHARDS="${COMPARE_N_SHARDS:-1}"
if [ "${N_SHARDS}" -gt 1 ]; then
  SBATCH_ARGS+=(--array="0-$((N_SHARDS - 1))")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs outputs_compare

echo "Submit compare eval (shards: ${N_SHARDS})"
echo "  models  : ${COMPARE_MODELS}"
echo "  ckpt    : ${COMPARE_CHECKPOINT}"
echo "  out     : ${SCRIPT_DIR}/outputs_compare/${COMPARE_RUN_NAME}"

JOB_ID=$(TRAIN_PI05_CYCLE_EVAL_CONFIG="${CONFIG_PATH}" N_SHARDS="${N_SHARDS}" \
  sbatch --parsable "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/compare.sbatch")
echo "  job     : ${JOB_ID}"

if [ "${N_SHARDS}" -gt 1 ]; then
  MERGE_ID=$(sbatch --parsable --dependency=afterany:"${JOB_ID}" \
    --job-name=pi05_cyc_cmp_merge \
    --partition="${EVAL_PARTITION}" --qos="${EVAL_QOS}" \
    --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output=logs/%x_%j.out --error=logs/%x_%j.err \
    --wrap="source '${PROJECT_ROOT}/.venv/bin/activate' && python '${ROOT_DIR}/src/merge_compare_summaries.py' --out_dir '${SCRIPT_DIR}/outputs_compare/${COMPARE_RUN_NAME}'")
  echo "  merge   : ${MERGE_ID} (afterany:${JOB_ID})"
fi
