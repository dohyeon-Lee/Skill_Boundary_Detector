#!/usr/bin/env bash
#SBATCH --job-name=FSQ_GT
# No --gres here: the replay only decodes dataset video on CPU. A #SBATCH
# default still applies when the submit line omits --gres, so leaving gpu:1
# here would silently make every task queue for a GPU it never uses.
# submit_fsq_gt_replay.sh passes --gres only when slurm.replay_gres is set.
#SBATCH --requeue
# submit_fsq_gt_replay.sh overrides these with slurm.replay_cpus/replay_memory;
# they are only the fallback for a hand-run sbatch.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${FSQ_GT_REPLAY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FSQ_GT_REPLAY_CONFIG:-${SUBMIT_DIR}/fsq_gt_replay_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

RUN_ARGS=()
[ -z "${FSQ_GT_REPLAY_RUN:-}" ] || RUN_ARGS=(--run-name "${FSQ_GT_REPLAY_RUN}")

if [ -n "${FSQ_GT_REPLAY_CHECKPOINTS:-}" ]; then
  read -r -a CHECKPOINTS <<< "${FSQ_GT_REPLAY_CHECKPOINTS}"
  WORKER_COUNT="${FSQ_GT_REPLAY_WORKER_COUNT:-1}"
  FLAT_INDEX="${FSQ_GT_REPLAY_ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  CHUNK="${FSQ_GT_REPLAY_CHUNK:-1}"
  CHUNK_INDEX=$((FLAT_INDEX / WORKER_COUNT))
  WORKER_INDEX=$((FLAT_INDEX % WORKER_COUNT))
  CHUNK_START=$((CHUNK_INDEX * CHUNK))
  if [ "${CHUNK_START}" -ge "${#CHECKPOINTS[@]}" ]; then
    echo "Invalid replay array index ${FLAT_INDEX}." >&2
    exit 1
  fi
  MY_CHECKPOINTS=("${CHECKPOINTS[@]:${CHUNK_START}:${CHUNK}}")
  # Resolve the first one for the settings shared by the whole chunk (roots,
  # dataset, task selection); the per-checkpoint paths go into the plan below.
  # Reuse the submit-time list rather than rescanning: training that keeps
  # writing checkpoints must not enlarge the set this array is waiting for.
  SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --checkpoint "${MY_CHECKPOINTS[0]}" \
      --expected-checkpoints "${FSQ_GT_REPLAY_CHECKPOINTS}" \
      ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} --shell
  )"
  eval "${SETTINGS}"
elif [ -z "${FSQ_MODEL_PATH:-}" ]; then
  SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} --shell
  )"
  eval "${SETTINGS}"
  WORKER_INDEX="${FSQ_GT_REPLAY_WORKER_INDEX:-0}"
  WORKER_COUNT="${FSQ_GT_REPLAY_WORKER_COUNT:-1}"
else
  WORKER_INDEX="${FSQ_GT_REPLAY_WORKER_INDEX:-0}"
  WORKER_COUNT="${FSQ_GT_REPLAY_WORKER_COUNT:-1}"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_CONFIG_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${LIBERO_CONFIG_TMP_ROOT}/fsq_gt_libero.XXXXXX")"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}" "${SUBMIT_DIR}/logs" "${EVAL_OUT_DIR}"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

# EVAL_INIT_STATES_PATH and ORIGINAL_DATASET_DIR are empty when episode_source
# is "dataset": episodes are then grouped by the dataset's own task table.
for artifact in \
  "${FSQ_MODEL_PATH}" \
  "${FSQ_SKILLS_DIR}" \
  "${SKILL_DATASET_DIR}" \
  "${EVAL_INIT_STATES_PATH}" \
  "${ORIGINAL_DATASET_DIR}"; do
  [ -n "${artifact}" ] || continue
  if [ ! -e "${artifact}" ]; then
    echo "Missing FSQ GT replay artifact: ${artifact}" >&2
    exit 1
  fi
done

# Build this task's replay plan. Every checkpoint in the chunk is replayed by
# ONE python process: start-up (torch + lerobot imports) costs minutes here
# while a checkpoint's replay costs seconds to minutes, so paying it per
# checkpoint is what made these jobs slow.
PLAN_FILE="$(mktemp "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/fsq_gt_plan.XXXXXX.json")"
trap 'rm -f "${PLAN_FILE}"' EXIT
PLAN_ENTRIES=()
for CHECKPOINT in "${MY_CHECKPOINTS[@]}"; do
  ENTRY_SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT}" \
      --expected-checkpoints "${FSQ_GT_REPLAY_CHECKPOINTS}" \
      ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} --shell
  )"
  eval "${ENTRY_SETTINGS}"
  # Latents come from the GPU prepass so this array can stay GPU-free; encoding
  # them here on CPU would take about an hour each, so fail fast instead.
  if [ ! -f "${FSQ_LATENTS_PATH}" ]; then
    echo "Missing FSQ latents: ${FSQ_LATENTS_PATH}" >&2
    echo "Run the latents prepass first (submit_fsq_gt_replay.sh does it)." >&2
    exit 1
  fi
  mkdir -p "${EVAL_OUT_DIR}"
  PLAN_ENTRIES+=("{\"model_path\": \"${FSQ_MODEL_PATH}\", \"latents_path\": \"${FSQ_LATENTS_PATH}\", \"output_dir\": \"${EVAL_OUT_DIR}\", \"epoch_tag\": \"${FSQ_EPOCH_TAG}\"}")
done
printf '[%s]\n' "$(IFS=,; echo "${PLAN_ENTRIES[*]}")" > "${PLAN_FILE}"

echo "START FSQ GT SKILL REPLAY"
echo "node      : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "worker    : ${WORKER_INDEX}/${WORKER_COUNT}"
echo "FSQ       : ${FSQ_RUN_NAME} ${MY_CHECKPOINTS[*]} (${#MY_CHECKPOINTS[@]} checkpoint(s))"
echo "suite     : ${TARGET_TASK} task_ids=${TASK_IDS}"
echo "episodes  : ${EPISODES_PER_TASK}/task selection=${EPISODE_SELECTION} explicit=${EPISODE_IDS}"
echo "ep source : ${EPISODE_SOURCE}"
echo "rollout   : GT actions only"
echo "output    : ${EVAL_OUT_DIR}"
echo "gpu       : ${EVAL_REPLAY_GRES:-none (replay is CPU-only)}"
[ -z "${EVAL_REPLAY_GRES}" ] || nvidia-smi || true

# Fail over to another GPU instead of silently running on CPU -- only when this
# array actually asked for one. The replay itself decodes dataset video on CPU.
if [ -n "${EVAL_REPLAY_GRES}" ]; then
  source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
  require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
fi

RESUME_ARGS=()
[ "${EVAL_RESUME}" != "true" ] || RESUME_ARGS=(--resume)
COMPARE_ARGS=()
[ -z "${EVAL_COMPARE_DIR}" ] || COMPARE_ARGS=(
  --compare-dir "${EVAL_COMPARE_DIR}"
  --compare-collection-dirs "${EVAL_COMPARE_COLLECTION_DIRS}"
)
# A GPU-free job gets no health check from gpu_guard, and a node that hangs it
# before the first log line would otherwise burn the whole time limit doing
# nothing. run_fsq_gt_replay.py logs "episode source" once start-up is done
# (about 90s even on a loaded login node), so a long silence means the node.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/stall_guard.sh"
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  run_with_startup_guard "${STALL_GUARD_TIMEOUT:-600}" "episode source" -- \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_fsq_gt_replay.py" \
    --plan-path "${PLAN_FILE}" \
    --skill-dataset-dir "${SKILL_DATASET_DIR}" \
    --eval-init-states-path "${EVAL_INIT_STATES_PATH}" \
    --original-dataset-dir "${ORIGINAL_DATASET_DIR}" \
    --target-task "${TARGET_TASK}" \
    --task-ids "${TASK_IDS}" \
    --episode-ids "${EPISODE_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --episode-selection "${EPISODE_SELECTION}" \
    --collection-dir "${EVAL_COLLECTION_DIR}" \
    --expected-epoch-tags "${FSQ_EXPECTED_EPOCH_TAGS}" \
    --run-name "${FSQ_RUN_NAME}" \
    --worker-index "${WORKER_INDEX}" \
    --worker-count "${WORKER_COUNT}" \
    --seed "${EVAL_SEED}" \
    ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${EVAL_COLLECTION_DIR}/index.html"
