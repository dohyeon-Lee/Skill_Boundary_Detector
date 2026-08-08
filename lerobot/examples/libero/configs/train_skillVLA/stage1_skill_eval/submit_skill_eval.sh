#!/usr/bin/env bash
# Submit policy x episode workers for the multi-policy skill evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_SKILL_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_skill_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
STAGE1_SKILL_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_skill_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${STAGE1_SKILL_EVAL_EXPORTS}"

SBATCH_ARGS=(
  --job-name="${STAGE1_SKILL_EVAL_JOB_NAME:-S1skill}"
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
echo "Submit Stage-1 multi-policy skill eval"
echo "  policies : ${MODEL_COUNT} (${ARCHITECTURE_LABEL})"
echo "  MAIN     : ${MAIN_TERMINATOR_LABEL} (${MAIN_TERMINATOR_VARIANT})"
echo "  display  : ${TERMINATOR_MODEL_LABEL} (${TERMINATOR_MODEL_VARIANT})"
echo "  display rule: ${TERMINATOR_MODEL_END_MODE} term=${TERMINATOR_MODEL_END_THRESHOLD} progress=${TERMINATOR_MODEL_PROGRESS_THRESHOLD}"
echo "  tasks    : ${TARGET_TASK} ${TASK_IDS}"
echo "  episodes : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
echo "  shift    : ±${TIME_SHIFT_OFFSET}"
echo "  GPUs     : ${EVAL_NUM_GPUS} policy x episode workers"
echo "  output   : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode     : one worker via srun in allocation ${SLURM_JOB_ID}"
  SKILL_EVAL_WORKER_INDEX=0 SKILL_EVAL_WORKER_COUNT=1 \
    STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS}" -le 1 ]; then
  echo "  mode     : one sbatch job"
  SKILL_EVAL_WORKER_INDEX=0 SKILL_EVAL_WORKER_COUNT=1 \
    STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_NUM_GPUS - 1))%${EVAL_NUM_GPUS}"
  echo "  mode     : Slurm array ${ARRAY_SPEC}"
  export SKILL_EVAL_WORKER_COUNT="${EVAL_NUM_GPUS}"
  STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
