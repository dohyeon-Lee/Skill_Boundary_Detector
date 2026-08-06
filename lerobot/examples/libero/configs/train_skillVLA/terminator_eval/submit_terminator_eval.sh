#!/usr/bin/env bash
# Submit one GPU job for the single-model skill-segment evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TERMINATOR_EVAL_CONFIG:-${SCRIPT_DIR}/terminator_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
TERMINATOR_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${TERMINATOR_EVAL_EXPORTS}"

SBATCH_ARGS=(
  --job-name="${TERMINATOR_EVAL_JOB_NAME:-TermEval}"
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit multi-terminator skill eval"
echo "  model    : ${MODEL_LABEL} (${ARCHITECTURE_LABEL})"
echo "  tasks    : ${TARGET_TASK} ${TASK_IDS}"
echo "  episodes : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
echo "  shift    : ±${TIME_SHIFT_OFFSET}"
echo "  GPUs     : ${EVAL_NUM_GPUS} occurrence workers"
echo "  output   : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode     : one worker via srun in allocation ${SLURM_JOB_ID}"
  SKILL_EVAL_WORKER_INDEX=0 SKILL_EVAL_WORKER_COUNT=1 \
    TERMINATOR_EVAL_DIR="${SCRIPT_DIR}" TERMINATOR_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS}" -le 1 ]; then
  echo "  mode     : one sbatch job"
  SKILL_EVAL_WORKER_INDEX=0 SKILL_EVAL_WORKER_COUNT=1 \
    TERMINATOR_EVAL_DIR="${SCRIPT_DIR}" TERMINATOR_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_NUM_GPUS - 1))%${EVAL_NUM_GPUS}"
  echo "  mode     : Slurm array ${ARRAY_SPEC}"
  export SKILL_EVAL_WORKER_COUNT="${EVAL_NUM_GPUS}"
  TERMINATOR_EVAL_DIR="${SCRIPT_DIR}" TERMINATOR_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
