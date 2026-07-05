#!/usr/bin/env bash
# Side-by-side comparison eval: N cycle-PT models, task-by-task incremental video stitching.
#
# Inputs:
#   models : cycle_eval_config.yaml compare_models ({model_dir, label} list) + compare_checkpoint
#   env    : LIBERO sim suite {eval_target_task}, task ids {eval_task_ids}
# Output:
#   ./outputs_compare/{compare_run_name}/videos/task{ID}_ep{E}.mp4  (진행 중에도 생김)
#   ./outputs_compare/{compare_run_name}/compare_summary.json       (task마다 갱신)
# Config:
#   ./cycle_eval_config.yaml
#SBATCH --job-name=pi05_cyc_cmp
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

# Multi-GPU: submitted as a SLURM array (submit_compare.sh N_SHARDS>1) — each array element
# takes task_ids[k::n] on its own GPU; a dependency job merges shard summaries at the end.
N_SHARDS="${N_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "START compare eval (shard ${SHARD}/${N_SHARDS})"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "models  : ${COMPARE_MODELS}"
echo "ckpt    : ${COMPARE_CHECKPOINT}"
echo "target  : ${EVAL_TARGET_TASK} ids=${EVAL_TASK_IDS} eps=${EVAL_N_EPISODES}"
nvidia-smi || true

OUT_DIR="${SUBMIT_DIR}/outputs_compare/${COMPARE_RUN_NAME}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" "${PYTHON_BIN}" \
  "${ROOT_DIR}/src/eval_compare.py" \
  --models "${COMPARE_MODELS}" \
  --models_root "${PI05_CYCLE_OUTPUTS_ROOT}" \
  --checkpoint "${COMPARE_CHECKPOINT}" \
  --suite "${EVAL_TARGET_TASK}" \
  --task_ids "${EVAL_TASK_IDS}" \
  --n_episodes "${EVAL_N_EPISODES}" \
  --init_state_offset "${EVAL_EPISODE_OFFSET}" \
  --n_action_steps "${EVAL_N_ACTION_STEPS}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --n_shards "${N_SHARDS}" \
  --shard "${SHARD}" \
  --out_dir "${OUT_DIR}"

echo "DONE -> ${OUT_DIR}"
