#!/usr/bin/env bash
# Submit the GT-skill probe of FSQ co-trained terminators (one task per model).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FSQ_TERM_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_terminator_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_terminator_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"

cd "${SCRIPT_DIR}"
mkdir -p logs outputs
read -r -a LABELS <<< "${FSQ_MODEL_LABELS}"
TOTAL_JOBS="${#LABELS[@]}"
CONCURRENT="$(( EVAL_NUM_GPUS < TOTAL_JOBS ? EVAL_NUM_GPUS : TOTAL_JOBS ))"

echo "Submit FSQ terminator probe"
echo "  models    : ${TOTAL_JOBS} (${FSQ_MODEL_LABELS})"
echo "  skillset  : ${FSQ_SKILLS_DIR}"
echo "  tasks     : ${TARGET_TASK} ${TASK_IDS}"
echo "  episodes  : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
echo "  threshold : ${EVAL_END_THRESHOLD}"
echo "  GPUs      : at most ${CONCURRENT} concurrent"
echo "  output    : ${EVAL_COLLECTION_DIR}/index.html"

SBATCH_ARGS=(
  --job-name=FSQ_TERM
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

export FSQ_TERM_MODEL_LABELS="${FSQ_MODEL_LABELS}"
if [ "${TOTAL_JOBS}" -le 1 ]; then
  FSQ_TERM_ARRAY_INDEX=0 \
    FSQ_TERM_EVAL_DIR="${SCRIPT_DIR}" FSQ_TERM_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/fsq_terminator_eval.sbatch"
else
  ARRAY_SPEC="0-$((TOTAL_JOBS - 1))%${CONCURRENT}"
  echo "  array     : ${ARRAY_SPEC}"
  FSQ_TERM_EVAL_DIR="${SCRIPT_DIR}" FSQ_TERM_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/fsq_terminator_eval.sbatch"
fi
