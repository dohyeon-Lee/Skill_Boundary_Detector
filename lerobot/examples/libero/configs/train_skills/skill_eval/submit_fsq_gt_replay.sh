#!/usr/bin/env bash
# Submit the FSQ-only, GT-action skill replay evaluator (one job array per run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FSQ_GT_REPLAY_CONFIG:-${SCRIPT_DIR}/fsq_gt_replay_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"
read -r -a RUN_NAMES <<< "${FSQ_EVAL_RUN_NAMES}"

cd "${SCRIPT_DIR}"
mkdir -p logs outputs
echo "Submit FSQ GT skill replay (${#RUN_NAMES[@]} run(s))"

for RUN_NAME in "${RUN_NAMES[@]}"; do
  SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --run-name "${RUN_NAME}" --shell
  )"
  eval "${SETTINGS}"
  read -r -a FSQ_CHECKPOINTS <<< "${FSQ_EVAL_CHECKPOINTS}"
  TOTAL_JOBS=$((FSQ_CHECKPOINT_COUNT * EVAL_WORKER_COUNT))
  export FSQ_GT_REPLAY_CHECKPOINTS="${FSQ_EVAL_CHECKPOINTS}"
  export FSQ_GT_REPLAY_WORKER_COUNT="${EVAL_WORKER_COUNT}"
  export FSQ_GT_REPLAY_RUN="${RUN_NAME}"

  # Encoding a checkpoint's skill latents is the only GPU work here, so it goes
  # into one prepass job per run and the replay array below asks for no GPU at
  # all. Three models to compare therefore need three GPUs, not one per
  # checkpoint -- and the replay array schedules against idle CPU nodes.
  DEPENDENCY_ARGS=()
  if [ -n "${FSQ_MISSING_LATENTS}" ]; then
    PREPASS_ID="$(
      FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}" FSQ_GT_REPLAY_CONFIG="${CONFIG_PATH}" \
        FSQ_GT_REPLAY_RUN="${RUN_NAME}" \
        FSQ_LATENTS_CHECKPOINTS="${FSQ_MISSING_LATENTS}" \
        sbatch --parsable \
          --job-name=FSQ_LAT \
          --partition="${EVAL_PARTITION}" \
          --qos="${EVAL_QOS}" \
          --gres="${EVAL_GRES}" \
          --cpus-per-task="${EVAL_CPUS_PER_TASK}" \
          --mem="${EVAL_MEM}" \
          --time="${EVAL_LATENTS_TIME}" \
          ${EVAL_NODELIST:+--nodelist="${EVAL_NODELIST}"} \
          ${EVAL_EXCLUDE_NODES:+--exclude="${EVAL_EXCLUDE_NODES}"} \
          --output=logs/%x_%j.out --error=logs/%x_%j.err \
          "${SRC_DIR}/fsq_latents_prepass.sbatch"
    )"
    echo "  latents  : ${FSQ_MISSING_LATENTS_COUNT} checkpoint(s) to encode -> GPU job ${PREPASS_ID}"
    DEPENDENCY_ARGS=(--dependency="afterok:${PREPASS_ID}")
  else
    echo "  latents  : all present, no GPU needed"
  fi

  SBATCH_ARGS=(
    --job-name=FSQ_GT
    --partition="${EVAL_PARTITION}"
    --qos="${EVAL_QOS}"
    --cpus-per-task="${EVAL_CPUS_PER_TASK}"
    --mem="${EVAL_MEM}"
    --time="${EVAL_TIME}"
  )
  [ -z "${EVAL_REPLAY_GRES}" ] || SBATCH_ARGS+=(--gres="${EVAL_REPLAY_GRES}")
  [ "${#DEPENDENCY_ARGS[@]}" -eq 0 ] || SBATCH_ARGS+=("${DEPENDENCY_ARGS[@]}")
  [ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
  [ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

  echo "  FSQ      : ${FSQ_RUN_NAME} checkpoints=${FSQ_EVAL_CHECKPOINTS}"
  [ -z "${FSQ_SKIPPED_CHECKPOINTS}" ] || \
    echo "  skipped  : not trained yet -> ${FSQ_SKIPPED_CHECKPOINTS}"
  echo "  tasks    : ${TARGET_TASK} ${TASK_IDS}"
  echo "  episodes : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
  echo "  jobs     : ${FSQ_CHECKPOINT_COUNT} checkpoints x ${EVAL_WORKER_COUNT} workers"
  echo "  replay   : ${EVAL_REPLAY_GRES:-CPU-only (no GPU requested)}"
  echo "  slots    : at most ${EVAL_MAX_CONCURRENT} concurrent jobs"
  echo "  output   : ${EVAL_COLLECTION_DIR}/index.html"

  if [ -n "${SLURM_JOB_ID:-}" ]; then
    STATUS=0
    PIDS=()
    for ((INDEX=0; INDEX<TOTAL_JOBS; INDEX++)); do
      FSQ_GT_REPLAY_ARRAY_INDEX="${INDEX}" \
        FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}" FSQ_GT_REPLAY_CONFIG="${CONFIG_PATH}" \
        srun --exclusive --nodes=1 --ntasks=1 \
          ${EVAL_REPLAY_GRES:+--gres="${EVAL_REPLAY_GRES}"} \
          "${SRC_DIR}/fsq_gt_replay.sbatch" &
      PIDS+=("$!")
      if [ "${#PIDS[@]}" -eq "${EVAL_MAX_CONCURRENT}" ] || [ "${INDEX}" -eq "$((TOTAL_JOBS - 1))" ]; then
        for PID in "${PIDS[@]}"; do wait "${PID}" || STATUS=1; done
        PIDS=()
      fi
    done
    [ "${STATUS}" -eq 0 ] || exit "${STATUS}"
  elif [ "${TOTAL_JOBS}" -le 1 ]; then
    FSQ_GT_REPLAY_ARRAY_INDEX=0 \
      FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}" FSQ_GT_REPLAY_CONFIG="${CONFIG_PATH}" \
      sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/fsq_gt_replay.sbatch"
  else
    ARRAY_SPEC="${FSQ_GT_REPLAY_ARRAY_SPEC:-0-$((TOTAL_JOBS - 1))%${EVAL_MAX_CONCURRENT}}"
    echo "  array    : ${ARRAY_SPEC}"
    FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}" FSQ_GT_REPLAY_CONFIG="${CONFIG_PATH}" \
      sbatch --array="${ARRAY_SPEC}" \
        --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
        "${SBATCH_ARGS[@]}" "${SRC_DIR}/fsq_gt_replay.sbatch"
  fi
done

if [ "${#RUN_NAMES[@]}" -gt 1 ]; then
  echo "Model comparison is built automatically when the last run finishes:"
  echo "  ${EVAL_COMPARE_DIR}/index.html"
fi
