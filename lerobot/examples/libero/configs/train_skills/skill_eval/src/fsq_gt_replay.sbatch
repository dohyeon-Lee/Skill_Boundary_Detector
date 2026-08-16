#!/usr/bin/env bash
#SBATCH --job-name=FSQ_GT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
  CHECKPOINT_INDEX=$((FLAT_INDEX / WORKER_COUNT))
  WORKER_INDEX=$((FLAT_INDEX % WORKER_COUNT))
  if [ "${CHECKPOINT_INDEX}" -ge "${#CHECKPOINTS[@]}" ]; then
    echo "Invalid replay array index ${FLAT_INDEX}." >&2
    exit 1
  fi
  CHECKPOINT_OVERRIDE="${CHECKPOINTS[${CHECKPOINT_INDEX}]}"
  # Reuse the submit-time list rather than rescanning: training that keeps
  # writing checkpoints must not enlarge the set this array is waiting for.
  SETTINGS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_gt_replay_config.py" \
      --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT_OVERRIDE}" \
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

if [ ! -f "${FSQ_LATENTS_PATH}" ]; then
  LATENTS_LOCK="${FSQ_LATENTS_PATH}.lock"
  (
    flock 9
    if [ ! -f "${FSQ_LATENTS_PATH}" ]; then
      TEMP_LATENTS="${FSQ_LATENTS_PATH%.npz}.tmp.${SLURM_JOB_ID:-$$}.npz"
      echo "FSQ latents missing; encoding this checkpoint only"
      echo "  checkpoint: ${FSQ_MODEL_PATH}"
      echo "  output    : ${FSQ_LATENTS_PATH}"
      PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero" \
        "${PROJECT_ROOT}/.venv/bin/python" \
          "${LEROBOT_ROOT}/examples/libero/encode_FSQ_skills.py" \
          --model-path "${FSQ_MODEL_PATH}" \
          --skills-dir "${FSQ_SKILLS_DIR}" \
          --output-path "${TEMP_LATENTS}" \
          --device cuda
      mv "${TEMP_LATENTS}" "${FSQ_LATENTS_PATH}"
    fi
  ) 9>"${LATENTS_LOCK}"
fi

echo "START FSQ GT SKILL REPLAY"
echo "node      : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "worker    : ${WORKER_INDEX}/${WORKER_COUNT}"
echo "FSQ       : ${FSQ_RUN_NAME} ${FSQ_EPOCH_TAG}"
echo "suite     : ${TARGET_TASK} task_ids=${TASK_IDS}"
echo "episodes  : ${EPISODES_PER_TASK}/task selection=${EPISODE_SELECTION} explicit=${EPISODE_IDS}"
echo "ep source : ${EPISODE_SOURCE}"
echo "rollout   : GT actions only"
echo "output    : ${EVAL_OUT_DIR}"
nvidia-smi || true

RESUME_ARGS=()
[ "${EVAL_RESUME}" != "true" ] || RESUME_ARGS=(--resume)
COMPARE_ARGS=()
[ -z "${EVAL_COMPARE_DIR}" ] || COMPARE_ARGS=(
  --compare-dir "${EVAL_COMPARE_DIR}"
  --compare-collection-dirs "${EVAL_COMPARE_COLLECTION_DIRS}"
)
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_fsq_gt_replay.py" \
    --model-path "${FSQ_MODEL_PATH}" \
    --latents-path "${FSQ_LATENTS_PATH}" \
    --skill-dataset-dir "${SKILL_DATASET_DIR}" \
    --eval-init-states-path "${EVAL_INIT_STATES_PATH}" \
    --original-dataset-dir "${ORIGINAL_DATASET_DIR}" \
    --target-task "${TARGET_TASK}" \
    --task-ids "${TASK_IDS}" \
    --episode-ids "${EPISODE_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --episode-selection "${EPISODE_SELECTION}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --collection-dir "${EVAL_COLLECTION_DIR}" \
    --expected-epoch-tags "${FSQ_EXPECTED_EPOCH_TAGS}" \
    --run-name "${FSQ_RUN_NAME}" \
    --epoch-tag "${FSQ_EPOCH_TAG}" \
    --worker-index "${WORKER_INDEX}" \
    --worker-count "${WORKER_COUNT}" \
    --seed "${EVAL_SEED}" \
    ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${EVAL_COLLECTION_DIR}/index.html"
