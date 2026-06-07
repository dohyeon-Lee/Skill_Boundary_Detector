#!/usr/bin/env bash
# SkillVLA Stage-1 (skill_expert) closed-loop ORACLE eval on LIBERO sim.
# GT skill sequence per task (matched by language) + frozen FSQ terminator advances skills.
#SBATCH --job-name=skillExpert_S1_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # stage1_eval
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SUBMIT_DIR}/stage1_eval_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

# ── LIBERO sim config (benchmark + bddl + init_states) ──
LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_LOCAL_CONFIG_DIR="${LIBERO_REPO_ROOT}/.libero_config"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START STAGE-1 ORACLE EVAL (skill_expert)"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "policy   : ${POLICY_PATH}"
echo "fsq      : ${FSQ_CKPT}"
echo "skills   : ${SKILL_LABEL_DATASET_DIR}"
echo "target   : ${TARGET_TASK}  task_ids=${TASK_IDS}  episodes=${N_EPISODES}/task"
nvidia-smi || true

for p in "${POLICY_PATH}" "${FSQ_CKPT}" "${SKILL_LABEL_DATASET_DIR}"; do
  if [ ! -e "${p}" ]; then
    echo "Missing required artifact: ${p}" >&2
    exit 1
  fi
done
mkdir -p "${EVAL_OUT_DIR}"

# skill_html needs skill_latents.npz; auto-disable if it was cleaned up.
SKILL_HTML_EFFECTIVE="${SKILL_HTML}"
if [ "${SKILL_HTML}" = "true" ] && [ ! -f "${SKILL_LATENTS_PATH}" ]; then
  echo "[warn] skill_latents.npz missing (${SKILL_LATENTS_PATH}); disabling skill_html."
  SKILL_HTML_EFFECTIVE=false
fi

# Optional args: wandb only when enabled.
EXTRA_ARGS=()
if [ "${WANDB_ENABLE}" = "true" ]; then
  EXTRA_ARGS+=(--wandb_project="${WANDB_PROJECT}")
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_eval.py" \
    --policy.path="${POLICY_PATH}" \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.skill_label_dataset_dir="${SKILL_LABEL_DATASET_DIR}" \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.skill_advance_mode="${SKILL_ADVANCE_MODE}" \
    --policy.skill_end_mode="${SKILL_END_MODE}" \
    --policy.skill_end_threshold="${SKILL_END_THRESHOLD}" \
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
    --eval.max_videos_per_task="${MAX_VIDEOS_PER_TASK}" \
    --eval.video_frame_stride="${VIDEO_FRAME_STRIDE}" \
    --eval.video_fps="${VIDEO_FPS}" \
    --eval.skill_html="${SKILL_HTML_EFFECTIVE}" \
    --eval.skill_html_train_samples="${SKILL_HTML_TRAIN_SAMPLES}" \
    --eval.skill_html_skill_latents_path="${SKILL_LATENTS_PATH}" \
    --eval.skill_html_raw_dataset_dir="${SKILL_HTML_RAW_DATASET_DIR}" \
    --eval.skill_html_image_key="${IMAGE_KEY}" \
    --output_dir="${EVAL_OUT_DIR}" \
    --job_name="${WANDB_RUN_NAME}" \
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "DONE → ${EVAL_OUT_DIR}"
