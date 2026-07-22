#!/usr/bin/env bash
# FSQ_new A/B/C closed-loop oracle eval on LIBERO.
#SBATCH --job-name=FSQNeval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${FSQ_NEW_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FSQ_NEW_EVAL_CONFIG:-${SUBMIT_DIR}/fsq_new_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${FSQ_NEW_EVAL_ENV_SNAPSHOT:-}" ] && [ -f "${FSQ_NEW_EVAL_ENV_SNAPSHOT}" ]; then
  source "${FSQ_NEW_EVAL_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_new_eval_config.py" --config "${CONFIG_PATH}" --shell)"
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_FANOUT:-}" ]; then
  LINE="$(printf '%s\n' "${EVAL_FANOUT}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  IFS='|' read -r PANEL_KEY TASK_IDS TASK_TAG <<< "${LINE}"
  export PANEL_KEY TASK_IDS TASK_TAG
  echo "array task ${SLURM_ARRAY_TASK_ID}: panel=${PANEL_KEY} tasks=${TASK_IDS} tag=${TASK_TAG}"
fi

PANEL_SPEC="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; key=os.environ["PANEL_KEY"]; print(json.dumps(next(x for x in json.loads(os.environ["MODELS_JSON"]) if x["key"]==key)))')"
export PANEL_SPEC
PANEL_LABEL="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(json.loads(os.environ["PANEL_SPEC"])["label"])')"
PANEL_OUT="${EVAL_OUT_DIR}/panels/${PANEL_KEY}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_LOCAL_CONFIG_DIR="${LIBERO_REPO_ROOT}/.libero_config"
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

FSQ_PATH="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(json.loads(os.environ["PANEL_SPEC"])["fsq_path"])')"
LATENTS_PATH="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(json.loads(os.environ["PANEL_SPEC"])["latents_path"])')"
for path in "${FSQ_PATH}" "${LATENTS_PATH}" "${EVAL_INIT_STATES_PATH}" "${RAW_DATASET_DIR}"; do
  [ -e "${path}" ] || { echo "Missing FSQ_new eval artifact: ${path}" >&2; exit 1; }
done

EXTRA_ARGS=()
if [ "${WANDB_ENABLE}" = "true" ]; then
  EXTRA_ARGS+=(--wandb_project="${WANDB_PROJECT}")
fi
JOB_NAME="${WANDB_RUN_NAME}_${PANEL_KEY}${TASK_TAG:+_${TASK_TAG}}"
echo "START FSQ_new EVAL: ${PANEL_LABEL} | tasks=${TASK_IDS} | checkpoint=${FSQ_PATH}"
nvidia-smi || true

PANEL_SPEC="${PANEL_SPEC}" TASK_TAG="${TASK_TAG:-}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src:${SRC_DIR}:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_eval.py" \
    --policy.type=skill_expert \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.skill_label_dataset_dir="${LATENTS_PATH}" \
    --policy.eval_init_states_path="${EVAL_INIT_STATES_PATH}" \
    --policy.terminator_dino_model_path="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(json.loads(os.environ["PANEL_SPEC"])["dino_model_path"])')" \
    --policy.skill_end_mode="${SKILL_END_MODE}" \
    --policy.skill_end_threshold="${SKILL_END_THRESHOLD}" \
    --policy.skill_end_progress_threshold="${SKILL_END_PROGRESS_THRESHOLD}" \
    --policy.inference_skill_max_length="${INFERENCE_SKILL_MAX_LENGTH}" \
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
    --eval.skill_html="${SKILL_HTML}" \
    --eval.skill_html_train_samples="${SKILL_HTML_TRAIN_SAMPLES}" \
    --output_dir="${PANEL_OUT}" \
    --job_name="${JOB_NAME}" \
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

BASE_JOB="${WANDB_RUN_NAME}_${PANEL_KEY}"
if [ -n "${TASK_TAG:-}" ]; then
  MERGE_WANDB_ARGS=()
  [ "${WANDB_ENABLE}" = "true" ] && MERGE_WANDB_ARGS+=(--wandb_project="${WANDB_PROJECT}")
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/merge_eval_chunks.py" \
    --out_dir="${PANEL_OUT}" --target_task="${TARGET_TASK}" --job_name="${BASE_JOB}" \
    --expected_tasks="${TASKS_TOTAL:-0}" \
    "${MERGE_WANDB_ARGS[@]}" \
    || echo "chunk merge failed (non-fatal)"
fi

"${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/stitch_panels.py" \
  --panels_root="${EVAL_OUT_DIR}/panels" --out_dir="${EVAL_OUT_DIR}/side_by_side" \
  --labels_json="${MODELS_LABELS}" --per_row="${MODELS_PER_ROW}" || true
"${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/compare_success.py" \
  --panels_root="${EVAL_OUT_DIR}/panels" \
  --out="${EVAL_OUT_DIR}/task_success_comparison.png" \
  --labels_json="${MODELS_LABELS}" || true

echo "DONE ${PANEL_LABEL} -> ${PANEL_OUT}"
