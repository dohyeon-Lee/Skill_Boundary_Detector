#!/usr/bin/env bash
# pi05 closed-loop evaluation on LIBERO (lerobot-eval). One Slurm element owns one GPU and runs its
# packed task workers concurrently; each worker evaluates its task ids for every panel.
#SBATCH --job-name=pi05eval
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${PI05_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${PI05_EVAL_CONFIG:-${SUBMIT_DIR}/pi05_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

# submit_eval.sh exports one frozen resolution. Resolve here only for a direct `sbatch src/eval.sbatch`.
if [ -z "${MODELS_JSON:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/pi05_eval_config.py" --config "${CONFIG_PATH}" --shell)"
fi
if [ -z "${EVAL_GPU_GROUPS_JSON:-}" ]; then
  eval "$(
    "${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../../src/eval_gpu_packing.py" \
      --items-json "${TASK_IDS}" --gpus 1 \
      --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" --shell
  )"
fi

cd "${LEROBOT_ROOT}"
SHARED_VENV="$(readlink -f "${PROJECT_ROOT}/.venv")"
EVAL_VENV="${SHARED_VENV}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/node_local_venv.sh"
if [ -n "${EVAL_VENV_ARCHIVE:-}" ]; then
  if STAGED_VENV="$(stage_node_local_venv "${EVAL_VENV_ARCHIVE}" "${SHARED_VENV}" "pi05 eval venv")"; then
    EVAL_VENV="${STAGED_VENV}"
  fi
fi
EVAL_PYTHON="${EVAL_VENV}/bin/python"
export VIRTUAL_ENV="${EVAL_VENV}"
export PATH="${EVAL_VENV}/bin:${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
# A private config dir per job: a shared file let one job read a config another was mid-rewrite.
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/libero_cfg.XXXXXX")"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

mapfile -t PANEL_ROWS < <(
  "${BOOTSTRAP_PYTHON}" - <<'PY'
import json, os
for panel in json.loads(os.environ["MODELS_JSON"]):
    print(panel["label"] + "\t" + panel["policy_path"] + "\t" + panel["panel_dir"])
PY
)
for row in "${PANEL_ROWS[@]}"; do
  IFS=$'\t' read -r _label _policy _panel_dir <<< "${row}"
  [ -d "${_policy}" ] || { echo "Missing pi05 checkpoint for panel '${_label}': ${_policy}" >&2; exit 1; }
done
mkdir -p "${EVAL_OUT_DIR}/panels" "${EVAL_OUT_DIR}/metrics" "${SUBMIT_DIR}/logs"

echo "START pi05 EVAL"
echo "node     : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "job      : ${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "cuda     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "venv     : ${EVAL_VENV}"
echo "panels   : ${PANEL_COUNT} -> ${PANEL_SUMMARY}"
echo "suite    : ${TARGET_TASK} task_ids=${TASK_IDS} episodes=${N_EPISODES}"
echo "init     : $([ "${EPISODE_EXACT}" = true ] && echo "episode-exact ${EVAL_INIT_STATES_PATH}" || echo "benchmark init states, offset=${EPISODE_OFFSET}")"
echo "video    : enable=${VIDEO_ENABLE} max_per_task=${MAX_VIDEOS_PER_TASK} wrist_panel=${VIDEO_SHOW_WRIST} grid_columns=${GRID_COLUMNS}"
echo "output   : ${EVAL_OUT_DIR} (resume=${EVAL_RESUME})"
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
    print(json.dumps(worker["items"], separators=(",", ":")) + "\t" + str(worker["tag"]))
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

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/stall_guard.sh"

# One worker = its task ids through every panel, sequentially (one model on the GPU at a time).
panel_done() {   # label tag -> 0 when that worker already recorded the panel (Stage-1 metrics file)
  "${BOOTSTRAP_PYTHON}" - "${EVAL_OUT_DIR}/metrics/eval_info_$2.json" "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
sys.exit(0 if path.is_file() and sys.argv[2] in json.loads(path.read_text() or "{}") else 1)
PY
}

run_worker() {
  local worker_task_ids="$1" worker_tag="$2" row label policy_path panel_dir out
  for row in "${PANEL_ROWS[@]}"; do
    IFS=$'\t' read -r label policy_path panel_dir <<< "${row}"
    out="${EVAL_OUT_DIR}/panels/${panel_dir}"
    mkdir -p "${out}"
    if [ "${EVAL_RESUME}" = true ] && panel_done "${label}" "${worker_tag}"; then
      echo "[${worker_tag}] resume: panel '${label}' already complete; skipping."
      # The startup guard still needs a marker when every panel of a worker is skipped.
      touch "${LEROBOT_STARTUP_READY_MARKER}"
      continue
    fi
    echo "[${worker_tag}] panel '${label}': ${policy_path}"
    TASK_TAG="${worker_tag}" PYTORCH_ALLOC_CONF=expandable_segments:True \
    LEROBOT_EVAL_ANNOTATE_VIDEOS=1 LEROBOT_EVAL_VIDEO_WRIST="$([ "${VIDEO_SHOW_WRIST}" = true ] && echo 1 || echo 0)" \
    LEROBOT_EVAL_METRICS_DIR="${EVAL_OUT_DIR}/metrics" LEROBOT_EVAL_PANEL_LABEL="${label}" \
    PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" \
      "${EVAL_PYTHON}" -m lerobot.scripts.lerobot_eval \
        --policy.path="${policy_path}" \
        --policy.tokenizer_path="${PI05_TOKENIZER_PATH}" \
        --policy.n_action_steps="${N_ACTION_STEPS}" \
        --policy.device=cuda \
        --policy.compile_model=false \
        --policy.gradient_checkpointing=false \
        --env.type=libero \
        --env.task="${TARGET_TASK}" \
        --env.task_ids="${worker_task_ids}" \
        --env.init_state_offset="${EPISODE_OFFSET}" \
        --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
        --eval.batch_size="${EVAL_BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.use_async_envs=false \
        --eval.max_videos_per_task="${MAX_VIDEOS_PER_TASK}" \
        --eval.video_frame_stride="${VIDEO_FRAME_STRIDE}" \
        --eval.video_fps="${VIDEO_FPS}" \
        --eval.init_states_path="${EVAL_INIT_STATES_PATH}" \
        --output_dir="${out}" \
        --job_name="${WANDB_RUN_NAME}_${label}_${worker_tag}" \
        --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'
  done
}

MODEL_LOAD_LOCK="$(mktemp "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/pi05eval_load.XXXXXX.lock")"
WORKER_PIDS=()
WORKER_LABELS=()
WORKER_MARKERS=()
WORKER_CUDA_MARKERS=()
WORKER_ERR_LOGS=()
for row in "${LOCAL_WORKER_ROWS[@]}"; do
  IFS=$'\t' read -r worker_task_ids worker_tag <<< "${row}"
  log_prefix="${SUBMIT_DIR}/logs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${GPU_GROUP_INDEX}_${worker_tag}"
  ready_marker="${SLURM_TMPDIR:-/tmp}/pi05eval_${SLURM_JOB_ID:-$$}_${worker_tag}.ready"
  cuda_marker="${SLURM_TMPDIR:-/tmp}/pi05eval_${SLURM_JOB_ID:-$$}_${worker_tag}.cuda_failed"
  rm -f "${ready_marker}" "${cuda_marker}"
  echo "launching ${worker_tag} tasks=${worker_task_ids}; logs=${log_prefix}.{out,err}"
  (
    export LEROBOT_STARTUP_READY_MARKER="${ready_marker}"
    export LEROBOT_CUDA_GUARD_FAILURE_MARKER="${cuda_marker}"
    # Packed workers load the 3B checkpoint one at a time (host-RAM peak); rollouts stay parallel.
    export LEROBOT_MODEL_LOAD_LOCK="${MODEL_LOAD_LOCK}"
    if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
      export LEROBOT_INLINE_CUDA_GUARD=0
    else
      export LEROBOT_INLINE_CUDA_GUARD=1
    fi
    export OMP_NUM_THREADS="${LOCAL_CPU_THREADS}"
    export MKL_NUM_THREADS="${LOCAL_CPU_THREADS}"
    run_worker "${worker_task_ids}" "${worker_tag}"
  ) >"${log_prefix}.out" 2>"${log_prefix}.err" &
  WORKER_PIDS+=("$!")
  WORKER_LABELS+=("${worker_tag}")
  WORKER_MARKERS+=("${ready_marker}")
  WORKER_CUDA_MARKERS+=("${cuda_marker}")
  WORKER_ERR_LOGS+=("${log_prefix}.err")
  sleep "${EVAL_LOCAL_WORKER_STAGGER_SECONDS:-2}"
done

# A node that hangs during imports never reaches its first log line; requeue instead of burning the limit.
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

# Array elements finish independently. Serialize the cheap merge/stitch so they cannot overwrite each
# other's temporary files; the last element to finish sees every chunk and leaves the complete result.
echo "MERGE pi05 EVAL METRICS"
WANDB_PROJECT_ARG=""
[ "${WANDB_ENABLE}" = true ] && WANDB_PROJECT_ARG="${WANDB_PROJECT}"
(
  flock -x 9
  "${EVAL_PYTHON}" "${SRC_DIR}/merge_eval_chunks.py" \
    --out_dir="${EVAL_OUT_DIR}" \
    --expected_tasks="${EVAL_EXPECTED_TASKS}" \
    --wandb_project="${WANDB_PROJECT_ARG}" \
    --job_name="${WANDB_RUN_NAME}" \
    || echo "metrics merge failed (non-fatal)"
  if [ "${VIDEO_ENABLE}" = true ] && [ "${PANEL_COUNT}" -gt 1 ]; then
    "${EVAL_PYTHON}" "${SRC_DIR}/stitch_panels.py" \
      --out_dir="${EVAL_OUT_DIR}" --grid_columns="${GRID_COLUMNS}" \
      || echo "side-by-side stitch failed (non-fatal)"
  fi
) 9>"${EVAL_OUT_DIR}/.merge_eval_chunks.lock"

echo "DONE -> ${EVAL_OUT_DIR}"
