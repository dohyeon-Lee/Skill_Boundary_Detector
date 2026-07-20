#!/usr/bin/env bash
# Cycle-PT closed-loop eval (+ per-group success aggregation).
#
# Inputs:
#   policy : {project_root}/{outputs_root}/pi05_cycle_PT/{eval_model}/checkpoints/{eval_checkpoint}/pretrained_model
#   groups : {..}/{eval_model}/groups.json  (per-group aggregation)
#   env    : LIBERO sim suite {eval_target_task}
# Output:
#   eval   : ./outputs/{eval_wandb_run_name}_offset{eval_episode_offset}/
#            (eval_info.json + group_success.json + group_success.png)
# Config:
#   ./cycle_eval_config.yaml
#SBATCH --job-name=pi05_cyc_eval
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"             # .../train_pi05_cycle/cycle_eval
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"          # .../train_pi05_cycle
CONFIG_PATH="${TRAIN_PI05_CYCLE_EVAL_CONFIG:-${SUBMIT_DIR}/cycle_eval_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_cycle_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

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

echo "START cycle-PT eval"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "target    : ${EVAL_TARGET_TASK}"
echo "task_ids  : ${EVAL_TASK_IDS}"
echo "episodes  : ${EVAL_N_EPISODES}"
echo "policy    : ${EVAL_POLICY_PATH}"
echo "groups    : ${EVAL_GROUPS_JSON}"
nvidia-smi || true

if [ ! -d "${EVAL_POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${EVAL_POLICY_PATH}" >&2
  exit 1
fi

EVAL_OUT_DIR="${EVAL_OUT_DIR:-${SUBMIT_DIR}/outputs/${EVAL_WANDB_RUN_NAME}_offset${EVAL_EPISODE_OFFSET}}"
mkdir -p "${EVAL_OUT_DIR}"

PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" "${EVAL_BIN}" \
  --policy.path="${EVAL_POLICY_PATH}" \
  --policy.tokenizer_path="${PI05_TOKENIZER_PATH}" \
  --env.type=libero \
  --env.task="${EVAL_TARGET_TASK}" \
  --eval.batch_size="${EVAL_BATCH_SIZE}" \
  --eval.n_episodes="${EVAL_N_EPISODES}" \
  --eval.max_videos_per_task="${EVAL_MAX_VIDEOS_PER_TASK}" \
  --eval.video_frame_stride="${EVAL_VIDEO_FRAME_STRIDE}" \
  --eval.video_fps="${EVAL_VIDEO_FPS}" \
  --policy.device=cuda \
  --policy.n_action_steps="${EVAL_N_ACTION_STEPS}" \
  --env.task_ids="${EVAL_TASK_IDS}" \
  --env.init_state_offset="${EVAL_EPISODE_OFFSET}" \
  --env.max_parallel_tasks="${EVAL_MAX_PARALLEL_TASKS}" \
  --wandb_project "${EVAL_WANDB_PROJECT}" \
  --job_name "${EVAL_WANDB_RUN_NAME}" \
  --output_dir="${EVAL_OUT_DIR}" \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

# ── per-group success aggregation (groups.json × LIBERO task language) ──
if [ -f "${EVAL_GROUPS_JSON}" ]; then
  PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" "${PYTHON_BIN}" \
    "${ROOT_DIR}/src/aggregate_group_success.py" \
    --eval_info "${EVAL_OUT_DIR}/eval_info.json" \
    --groups_json "${EVAL_GROUPS_JSON}" \
    --suite "${EVAL_TARGET_TASK}" \
    --out_dir "${EVAL_OUT_DIR}"
else
  echo "groups.json not found (${EVAL_GROUPS_JSON}) — skipping per-group aggregation." >&2
fi

echo "DONE -> ${EVAL_OUT_DIR}"
