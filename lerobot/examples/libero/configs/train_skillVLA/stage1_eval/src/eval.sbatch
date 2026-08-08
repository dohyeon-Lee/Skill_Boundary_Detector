#!/usr/bin/env bash
# Stage-1 VSA/Cond-Gemma multi-model closed-loop evaluation.
#SBATCH --job-name=S1eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${STAGE1_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SUBMIT_DIR}/stage1_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

# submit_eval.sh exports one frozen resolution.  Resolve here only for direct
# `sbatch src/eval.sbatch` use; this keeps array elements on the same output path.
if [ -z "${POLICY_PATH:-}" ]; then
  STAGE1_EVAL_EXPORTS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" \
      --config "${CONFIG_PATH}" --shell
  )"
  eval "${STAGE1_EVAL_EXPORTS}"
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_FANOUT:-}" ]; then
  LINE="$(printf '%s\n' "${EVAL_FANOUT}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  IFS='|' read -r TASK_IDS TASK_TAG <<< "${LINE}"
  export TASK_IDS TASK_TAG
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
LIBERO_LOCAL_CONFIG_DIR="${LIBERO_REPO_ROOT}/.libero_config"
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

for artifact in "${POLICY_PATH}" "${SKILL_DATASET_DIR}" "${DINO_MODEL_PATH}"; do
  if [ ! -e "${artifact}" ]; then
    echo "Missing Stage-1 eval artifact: ${artifact}" >&2
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

echo "START STAGE-1 EVAL"
echo "node     : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "job      : ${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "cuda     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "grid     : ${MODEL_COUNT} models x ${CHECKPOINT_COUNT} checkpoints = ${PANEL_COUNT} panels"
echo "models   : ${MODEL_ARCHITECTURES}"
echo "contract : auto-detected independently from every checkpoint"
echo "skills   : ${SKILL_DATASET_DIR}"
echo "term     : primary=${TERMINATOR_VARIANT} (used by advance_mode=external|own|original)"
echo "replan   : n_action_steps=${N_ACTION_STEPS} immediate_on_skill_end=${IMMEDIATE_REPLAN_ON_SKILL_END}"
echo "suite    : ${TARGET_TASK} task_ids=${TASK_IDS} episodes=${N_EPISODES}"
echo "output   : ${EVAL_OUT_DIR}"
nvidia-smi || true

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

echo "DONE -> ${EVAL_OUT_DIR}"
