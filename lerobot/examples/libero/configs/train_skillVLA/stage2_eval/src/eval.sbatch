#!/usr/bin/env bash
# Clean Stage-2 multi-model closed-loop evaluation.
#SBATCH --job-name=S2eval
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${STAGE2_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE2_EVAL_CONFIG:-${SUBMIT_DIR}/stage2_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

# submit_eval.sh exports one frozen resolution. Resolve only for direct sbatch use.
if [ -z "${POLICY_PATH:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" --config "${CONFIG_PATH}" --shell)"
fi

EVAL_ALL_TASK_IDS="${TASK_IDS}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_FANOUT:-}" ]; then
  LINE="$(printf '%s\n' "${EVAL_FANOUT}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  IFS='|' read -r TASK_IDS TASK_TAG PANEL_INDICES <<< "${LINE}"
  export TASK_IDS TASK_TAG
  # Third field selects one panel per unit; older two-field snapshots keep the
  # all-panels-per-job behavior.
  if [ -n "${PANEL_INDICES:-}" ]; then export PANEL_INDICES; fi
  WANDB_RUN_NAME="${WANDB_RUN_NAME}_${TASK_TAG}"
  export WANDB_RUN_NAME
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
# Every task gets its own config dir. Sharing one file across array tasks
# (and across concurrent libero jobs) let a task read a config another job
# was mid-rewrite, resolving bddl_files to a stale root and killing the run.
LIBERO_CONFIG_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${LIBERO_CONFIG_TMP_ROOT}/libero_cfg.XXXXXX")"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}" logs
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

for artifact in "${POLICY_PATH}" "${FSQ_PATH}" "${SKILL_DATASET_DIR}" "${DINO_MODEL_PATH}" "${TOKENIZER_PATH}"; do
  if [ ! -e "${artifact}" ]; then
    echo "Missing Stage-2 eval artifact: ${artifact}" >&2
    exit 1
  fi
done
for overlay in "${EXTERNAL_PREDICTOR_MODEL}" "${EXTERNAL_TERMINATOR_MODEL}"; do
  if [ -n "${overlay}" ] && [ ! -d "${overlay}" ]; then
    echo "Missing Stage-2 eval overlay checkpoint: ${overlay}" >&2
    exit 1
  fi
done
if [ -n "${EVAL_INIT_STATES_PATH}" ] && [ ! -f "${EVAL_INIT_STATES_PATH}" ]; then
  echo "Missing episode-exact init states: ${EVAL_INIT_STATES_PATH}" >&2
  exit 1
fi
mkdir -p "${EVAL_OUT_DIR}"

SKILL_HTML_EFFECTIVE="${SKILL_HTML}"
if [ "${SKILL_HTML}" = true ] && [ ! -f "${SKILL_LATENTS_PATH}" ]; then
  echo "[warn] ${SKILL_LATENTS_PATH} missing; disabling skill HTML."
  SKILL_HTML_EFFECTIVE=false
fi

WANDB_ARGS=()
if [ "${WANDB_ENABLE}" = true ]; then
  WANDB_ARGS+=(--wandb_project="${WANDB_PROJECT}")
fi

echo "START STAGE-2 EVAL"
echo "panels   : ${MODEL_COUNT} (primary=${POLICY_PATH})"
echo "skills   : ${SKILL_DATASET_DIR}"
echo "predictor: ${EXTERNAL_PREDICTOR_MODEL:-<none>}"
echo "term     : ${EXTERNAL_TERMINATOR_MODEL:-<none>}"
echo "suite    : ${TARGET_TASK} task_ids=${TASK_IDS} episodes=${N_EPISODES}"
echo "output   : ${EVAL_OUT_DIR}"
nvidia-smi || true

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_eval.py" \
    --policy.path="${POLICY_PATH}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.device=cuda \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --env.type=libero \
    --env.task="${TARGET_TASK}" \
    --env.fps=20 \
    --env.task_ids="${TASK_IDS}" \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.use_async_envs=false \
    --eval.max_videos_per_task="${MAX_VIDEOS_PER_TASK}" \
    --eval.video_frame_stride="${VIDEO_FRAME_STRIDE}" \
    --eval.video_fps="${VIDEO_FPS}" \
    --eval.skill_html="${SKILL_HTML_EFFECTIVE}" \
    --eval.skill_html_train_samples="${SKILL_HTML_TRAIN_SAMPLES}" \
    --eval.skill_html_skill_latents_path="${SKILL_LATENTS_PATH}" \
    --eval.skill_html_raw_dataset_dir="${RAW_DATASET_DIR}" \
    --eval.skill_html_image_key=observation.images.image \
    --output_dir="${EVAL_OUT_DIR}" \
    --job_name="${WANDB_RUN_NAME}" \
    --rename_map='{"observation.images.image2":"observation.images.wrist_image"}' \
    "${WANDB_ARGS[@]}"

# Merge every chunk finished so far and regenerate the success-rate chart.
# Idempotent: whichever array job ends last leaves the complete chart behind.
EXPECTED_TASKS="$("${BOOTSTRAP_PYTHON}" -c 'import json, sys; print(len(json.loads(sys.argv[1])))' "${EVAL_ALL_TASK_IDS}")"
"${PROJECT_ROOT}/.venv/bin/python" \
  "${SUBMIT_DIR}/../stage1_eval/src/merge_eval_chunks.py" \
  --out_dir="${EVAL_OUT_DIR}" --expected_tasks="${EXPECTED_TASKS}" \
  || echo "chunk merge failed (non-fatal)"

echo "DONE -> ${EVAL_OUT_DIR}"
