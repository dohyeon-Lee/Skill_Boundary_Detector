#!/usr/bin/env bash
#SBATCH --job-name=S1skill
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${STAGE1_SKILL_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_SKILL_EVAL_CONFIG:-${SUBMIT_DIR}/stage1_skill_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -z "${POLICY_PATH:-}" ]; then
  STAGE1_SKILL_EVAL_EXPORTS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_skill_eval_config.py" \
      --config "${CONFIG_PATH}" --shell
  )"
  eval "${STAGE1_SKILL_EVAL_EXPORTS}"
fi
export SKILL_EVAL_WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-${SKILL_EVAL_WORKER_INDEX:-0}}"
export SKILL_EVAL_WORKER_COUNT="${SKILL_EVAL_WORKER_COUNT:-1}"

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

for artifact in \
  "${POLICY_PATH}" \
  "${FSQ_PATH}" \
  "${TERMINATOR_MODEL_PATH}" \
  "${SKILL_DATASET_DIR}" \
  "${SKILL_LATENTS_PATH}" \
  "${EVAL_INIT_STATES_PATH}" \
  "${ORIGINAL_DATASET_DIR}" \
  "${DINO_MODEL_PATH}"; do
  if [ ! -e "${artifact}" ]; then
    echo "Missing Stage-1 skill eval artifact: ${artifact}" >&2
    exit 1
  fi
done
mkdir -p "${EVAL_OUT_DIR}"

echo "START STAGE-1 MULTI-POLICY SKILL EVAL"
echo "node      : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "worker    : ${SKILL_EVAL_WORKER_INDEX}/${SKILL_EVAL_WORKER_COUNT}"
echo "cuda      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "policies  : ${MODEL_COUNT} (${ARCHITECTURE_LABEL})"
echo "MAIN      : ${MAIN_TERMINATOR_LABEL} (${MAIN_TERMINATOR_VARIANT}) ${MAIN_TERMINATOR_PATH}"
echo "display   : ${TERMINATOR_MODEL_LABEL} (${TERMINATOR_MODEL_VARIANT}) ${TERMINATOR_MODEL_PATH}"
echo "display rule: mode=${TERMINATOR_MODEL_END_MODE} term=${TERMINATOR_MODEL_END_THRESHOLD} progress=${TERMINATOR_MODEL_PROGRESS_THRESHOLD}"
echo "suite     : ${TARGET_TASK} task_ids=${TASK_IDS}"
echo "episodes  : ${EPISODES_PER_TASK}/task selection=${EPISODE_SELECTION} explicit=${EPISODE_IDS}"
echo "exact     : ${EPISODE_EXACT} (mandatory)"
echo "branches  : GT | policy@0 | policy@0 alt-noise | policy@-${TIME_SHIFT_OFFSET} | policy@+${TIME_SHIFT_OFFSET}"
echo "stopping  : MAIN-only mode=${SKILL_END_MODE} max=${SKILL_MAX_LENGTH_MODE}:${SKILL_MAX_LENGTH_SCALE} finish_chunk=${FINISH_ACTION_CHUNK_ON_END}"
echo "output    : ${EVAL_OUT_DIR}"
nvidia-smi || true

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_skill_eval.py" \
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
    --env.task_ids="${TASK_IDS}" \
    --env.init_states=false \
    --env.fps=20 \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes="${EPISODES_PER_TASK}" \
    --eval.use_async_envs=false \
    --eval.max_videos_per_task=0 \
    --output_dir="${EVAL_OUT_DIR}" \
    --job_name="S1skill_${MODEL_LABEL}" \
    --seed="${EVAL_SEED}" \
    --rename_map='{"observation.images.image2":"observation.images.wrist_image"}'

echo "DONE -> ${EVAL_OUT_DIR}/index.html"
