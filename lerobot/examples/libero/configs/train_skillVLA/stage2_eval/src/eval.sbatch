#!/usr/bin/env bash
# Stage-2 multi-model closed-loop evaluation.
# Every array element owns one GPU and runs several task/panel workers on it.
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
PROJECT_ROOT_HINT="$(cd "${SUBMIT_DIR}/../../../../../.." && pwd)"
select_bootstrap_python() {
  local candidate resolved
  for candidate in \
    "${STAGE2_EVAL_BOOTSTRAP_PYTHON:-}" \
    "${PROJECT_ROOT_HINT}/.venv/bin/python" \
    python3.12 python3.11 python3.10 python3 /usr/bin/python3; do
    [ -n "${candidate}" ] || continue
    if [[ "${candidate}" == */* ]]; then
      [ -x "${candidate}" ] || continue
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
      [ -n "${resolved}" ] || continue
    fi
    if "${resolved}" -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
      >/dev/null 2>&1; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  echo "Stage-2 eval requires Python >= 3.10; no compatible interpreter found." >&2
  return 1
}
BOOTSTRAP_PYTHON="$(select_bootstrap_python)"

if [ -z "${POLICY_PATH:-}" ]; then
  STAGE2_EVAL_EXPORTS="$(
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" \
      --config "${CONFIG_PATH}" --shell
  )"
  eval "${STAGE2_EVAL_EXPORTS}"
fi

if [ -z "${EVAL_GPU_GROUPS_JSON:-}" ]; then
  PACKING_EXPORTS="$(
    "${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../eval_gpu_packing.py" \
      --items-json "${TASK_IDS}" \
      --panel-count "${MODEL_COUNT}" \
      --gpus 1 \
      --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" \
      --shell
  )"
  eval "${PACKING_EXPORTS}"
fi

cd "${LEROBOT_ROOT}"
SHARED_VENV="$(readlink -f "${PROJECT_ROOT}/.venv")"
EVAL_VENV="${SHARED_VENV}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
if [ -n "${EVAL_VENV_ARCHIVE:-}" ]; then
  if STAGED_VENV="$(stage_node_local_venv \
    "${EVAL_VENV_ARCHIVE}" "${SHARED_VENV}" "Stage-2 eval venv")"; then
    EVAL_VENV="${STAGED_VENV}"
  fi
fi
EVAL_PYTHON="${EVAL_VENV}/bin/python"
export VIRTUAL_ENV="${EVAL_VENV}"
export PATH="${EVAL_VENV}/bin:${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
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
echo "node     : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "job      : ${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "cuda     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "venv     : ${EVAL_VENV}"
echo "grid     : ${MODEL_COUNT} stage2/prior panels"
echo "suite    : ${TARGET_TASK} dataset_task_ids=${DATASET_TASK_IDS} env_task_ids=${TASK_IDS} episodes=${N_EPISODES}"
echo "output   : ${EVAL_OUT_DIR}"
nvidia-smi || true

GPU_GROUP_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
mapfile -t LOCAL_WORKER_ROWS < <(
  "${BOOTSTRAP_PYTHON}" - "${EVAL_GPU_GROUPS_JSON}" "${GPU_GROUP_INDEX}" <<'PY'
import json, sys
groups = json.loads(sys.argv[1])
index = int(sys.argv[2])
if not 0 <= index < len(groups):
    raise SystemExit(f"GPU group {index} outside [0, {len(groups)})")
for worker in groups[index]:
    print(
        str(int(worker["worker_index"]))
        + "\t" + json.dumps(worker["items"], separators=(",", ":"))
        + "\t" + ",".join(str(value) for value in worker["panels"])
        + "\t" + str(worker["tag"])
    )
PY
)
if [ "${#LOCAL_WORKER_ROWS[@]}" -eq 0 ]; then
  echo "No logical workers assigned to GPU group ${GPU_GROUP_INDEX}." >&2
  exit 1
fi
LOCAL_WORKER_COUNT="${#LOCAL_WORKER_ROWS[@]}"
LOCAL_CPU_THREADS=$((EVAL_CPUS_PER_TASK / LOCAL_WORKER_COUNT))
(( LOCAL_CPU_THREADS > 0 )) || LOCAL_CPU_THREADS=1
echo "workers  : ${LOCAL_WORKER_COUNT} local / ${EVAL_LOGICAL_WORKER_COUNT} total"
echo "CPU split: ${LOCAL_CPU_THREADS} threads x ${LOCAL_WORKER_COUNT} workers"

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/stall_guard.sh"

WORKER_PIDS=()
WORKER_LABELS=()
WORKER_MARKERS=()
WORKER_CUDA_MARKERS=()
WORKER_ERR_LOGS=()
for row in "${LOCAL_WORKER_ROWS[@]}"; do
  IFS=$'\t' read -r worker_index worker_task_ids worker_panel_indices worker_tag <<< "${row}"
  log_prefix="${SUBMIT_DIR}/logs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${GPU_GROUP_INDEX}_${worker_tag}"
  ready_marker="${SLURM_TMPDIR:-/tmp}/s2eval_${SLURM_JOB_ID:-$$}_${worker_tag}.ready"
  cuda_marker="${SLURM_TMPDIR:-/tmp}/s2eval_${SLURM_JOB_ID:-$$}_${worker_tag}.cuda_failed"
  rm -f "${ready_marker}" "${cuda_marker}"
  echo "launching ${worker_tag} tasks=${worker_task_ids} panels=${worker_panel_indices}; logs=${log_prefix}.{out,err}"
  (
    export TASK_IDS="${worker_task_ids}"
    export TASK_TAG="${worker_tag}"
    export PANEL_INDICES="${worker_panel_indices}"
    export LEROBOT_STARTUP_READY_MARKER="${ready_marker}"
    export LEROBOT_CUDA_GUARD_FAILURE_MARKER="${cuda_marker}"
    if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
      export LEROBOT_INLINE_CUDA_GUARD=0
    else
      export LEROBOT_INLINE_CUDA_GUARD=1
    fi
    export OMP_NUM_THREADS="${LOCAL_CPU_THREADS}"
    export MKL_NUM_THREADS="${LOCAL_CPU_THREADS}"
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
      "${EVAL_PYTHON}" "${SRC_DIR}/run_eval.py" \
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
        --env.terminate_on_success=false \
        --env.task_ids="${worker_task_ids}" \
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
        --job_name="${WANDB_RUN_NAME}_${worker_tag}" \
        --rename_map='{"observation.images.image2":"observation.images.wrist_image"}' \
        "${WANDB_ARGS[@]}"
  ) >"${log_prefix}.out" 2>"${log_prefix}.err" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("${worker_tag}")
  WORKER_MARKERS+=("${ready_marker}")
  WORKER_CUDA_MARKERS+=("${cuda_marker}")
  WORKER_ERR_LOGS+=("${log_prefix}.err")
  sleep "${EVAL_LOCAL_WORKER_STAGGER_SECONDS:-2}"
done

startup_timeout="${EVAL_STARTUP_TIMEOUT:-600}"
startup_elapsed=0
while (( startup_elapsed < startup_timeout )); do
  pending=0
  for index in "${!WORKER_PIDS[@]}"; do
    [ -f "${WORKER_MARKERS[index]}" ] && continue
    process_state="$(ps -o stat= -p "${WORKER_PIDS[index]}" 2>/dev/null | tr -d ' ')"
    if [ -n "${process_state}" ] && [[ "${process_state}" != Z* ]]; then
      pending=1
    fi
  done
  (( pending == 1 )) || break
  sleep 5
  startup_elapsed=$((startup_elapsed + 5))
done
if (( startup_elapsed >= startup_timeout )); then
  echo "STARTUP GUARD: at least one worker did not finish imports in ${startup_timeout}s." >&2
  for pid in "${WORKER_PIDS[@]}"; do
    _stall_guard_kill_tree "${pid}" TERM
  done
  guard_requeue_current_job "STARTUP GUARD" \
    "evaluation imports did not complete within ${startup_timeout}s" \
    "Exclude the node and resubmit."
fi

worker_status=0
cuda_failed=false
for index in "${!WORKER_PIDS[@]}"; do
  status=0
  wait "${WORKER_PIDS[index]}" || status=$?
  if [ -f "${WORKER_CUDA_MARKERS[index]}" ] || [ "${status}" -eq 86 ]; then
    cuda_failed=true
  fi
  if [ "${status}" -ne 0 ]; then
    worker_status="${status}"
    echo "worker ${WORKER_LABELS[index]} failed (${status}); stderr tail:" >&2
    tail -40 "${WORKER_ERR_LOGS[index]}" >&2 || true
  fi
done
if [ "${cuda_failed}" = true ]; then
  _gpu_guard_requeue_current_job
fi
if [ "${worker_status}" -ne 0 ]; then
  exit "${worker_status}"
fi

# The panel evaluators already preserve the two independent panel videos and
# regenerate side_by_side clips whenever both stage2/prior inputs exist. Only
# the scalar report merge needs serialization across concurrent workers.
echo "MERGE STAGE-2 EVAL METRICS"
(
  flock -x 9
  "${EVAL_PYTHON}" "${SUBMIT_DIR}/../stage1_eval/src/merge_eval_chunks.py" \
    --out_dir="${EVAL_OUT_DIR}" \
    --expected_tasks="${EVAL_EXPECTED_TASKS}"
) 9>"${EVAL_OUT_DIR}/.merge_eval_chunks.lock"

echo "DONE -> ${EVAL_OUT_DIR}"
