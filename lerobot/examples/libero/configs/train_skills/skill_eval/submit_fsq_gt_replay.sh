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
  # The resolver emits EVAL_MAX_CONCURRENT after clamping it to the pending
  # work count.  Do not feed that derived value back into the next resolver
  # call: an already-complete run legitimately emits zero, while the input
  # YAML concurrency must remain positive.
  env -u EVAL_MAX_CONCURRENT \
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
    env -u EVAL_MAX_CONCURRENT \
      "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --run-name "${RUN_NAME}" --shell
  )"
  eval "${SETTINGS}"
  if [ "${FSQ_PENDING_CHECKPOINT_COUNT}" -eq 0 ]; then
    echo "  FSQ      : ${FSQ_RUN_NAME} already complete; no job submitted"
    echo "  complete : ${FSQ_COMPLETED_CHECKPOINTS}"
    echo "  output   : ${EVAL_COLLECTION_DIR}/index.html"
    continue
  fi
  read -r -a FSQ_CHECKPOINTS <<< "${FSQ_PENDING_CHECKPOINTS}"
  # One array task replays a CHUNK of checkpoints in a single process, so the
  # multi-minute torch/lerobot import is paid once per chunk instead of once per
  # checkpoint.
  CHUNK="${EVAL_CHECKPOINTS_PER_JOB}"
  PENDING_COUNT="${#FSQ_CHECKPOINTS[@]}"
  if [ "${PENDING_COUNT}" -ne "${FSQ_PENDING_CHECKPOINT_COUNT}" ]; then
    echo "Pending checkpoint count mismatch: list=${PENDING_COUNT}, resolver=${FSQ_PENDING_CHECKPOINT_COUNT}." >&2
    exit 1
  fi
  # The resolver owns this calculation.  In particular, do not use
  # FSQ_CHECKPOINT_COUNT here: it includes completed checkpoints and used to
  # create invalid trailing array tasks whenever resume left only one pending.
  TOTAL_JOBS="${EVAL_TOTAL_JOBS}"
  EXPECTED_TOTAL_JOBS=$(( ((PENDING_COUNT + CHUNK - 1) / CHUNK) * EVAL_WORKER_COUNT ))
  if [ "${TOTAL_JOBS}" -ne "${EXPECTED_TOTAL_JOBS}" ]; then
    echo "Replay job count mismatch: resolver=${TOTAL_JOBS}, expected=${EXPECTED_TOTAL_JOBS}." >&2
    exit 1
  fi
  export FSQ_GT_REPLAY_CHECKPOINTS="${FSQ_PENDING_CHECKPOINTS}"
  export FSQ_GT_REPLAY_WORKER_COUNT="${EVAL_WORKER_COUNT}"
  export FSQ_GT_REPLAY_CHUNK="${CHUNK}"
  export FSQ_GT_REPLAY_RUN="${RUN_NAME}"

  # Encoding a checkpoint's skill latents is the only GPU computation here, so
  # it goes into one prepass job per run. Replay may still reserve a GPU when it
  # inherits a GPU QOS, although its computation remains CPU-only.
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

  # The resolver has already selected inherited train placement or an explicit
  # replay override, together with the corresponding GRES policy.
  REPLAY_PARTITION="${EVAL_REPLAY_PARTITION}"
  REPLAY_QOS="${EVAL_REPLAY_QOS}"
  SBATCH_ARGS=(
    --job-name=FSQ_GT
    --partition="${REPLAY_PARTITION}"
    --qos="${REPLAY_QOS}"
    --cpus-per-task="${EVAL_REPLAY_CPUS}"
    --mem="${EVAL_REPLAY_MEM}"
    --time="${EVAL_TIME}"
  )
  [ -z "${EVAL_REPLAY_GRES}" ] || SBATCH_ARGS+=(--gres="${EVAL_REPLAY_GRES}")
  [ "${#DEPENDENCY_ARGS[@]}" -eq 0 ] || SBATCH_ARGS+=("${DEPENDENCY_ARGS[@]}")
  [ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
  [ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

  echo "  FSQ      : ${FSQ_RUN_NAME} pending=${FSQ_PENDING_CHECKPOINTS}"
  [ -z "${FSQ_COMPLETED_CHECKPOINTS}" ] || \
    echo "  complete : ${FSQ_COMPLETED_CHECKPOINTS} (skipped)"
  [ -z "${FSQ_SKIPPED_CHECKPOINTS}" ] || \
    echo "  skipped  : not trained yet -> ${FSQ_SKIPPED_CHECKPOINTS}"
  echo "  tasks    : ${TARGET_TASK} ${TASK_IDS}"
  echo "  episodes : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
  echo "  jobs     : ${FSQ_PENDING_CHECKPOINT_COUNT} checkpoints / ${CHUNK} per task x ${EVAL_WORKER_COUNT} workers = ${TOTAL_JOBS}"
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
