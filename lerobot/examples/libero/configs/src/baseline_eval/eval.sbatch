#!/usr/bin/env bash
#SBATCH --job-name=baselineeval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT must be exported by submit_eval.sh}"
# sbatch copies this script to /var/spool/slurmd/.../slurm_script. BASH_SOURCE
# therefore names the spool directory, NOT the original evaluation sources.
SHARED_DIR="${PROJECT_ROOT}/lerobot/examples/libero/configs/src/baseline_eval"
if [[ ! -f "${SHARED_DIR}/run_eval.py" ]]; then
  echo "Missing baseline evaluation source: ${SHARED_DIR}/run_eval.py" >&2
  exit 2
fi
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${MODEL_PYTHON}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/tools/lerobot-libero"
unset LD_LIBRARY_PATH
# Create these before workers start: a startup failure must not cause another
# 'missing .merge_eval_chunks.lock' error that hides the original traceback.
mkdir -p "${EVAL_OUT_DIR}/metrics" "${EVAL_WORK_DIR}/logs"
GROUP="${BASELINE_GPU_GROUP:-${SLURM_ARRAY_TASK_ID:-0}}"
mapfile -t WORKERS < <("${SIM_PYTHON}" - "${EVAL_GPU_GROUPS_JSON}" "${GROUP}" <<'PY'
import json, sys
for worker in json.loads(sys.argv[1])[int(sys.argv[2])]:
    print(json.dumps(worker["items"], separators=(",", ":")) + "\t" + worker["tag"])
PY
)
(( ${#WORKERS[@]} > 0 )) || { echo "No tasks assigned" >&2; exit 1; }
THREADS=$((EVAL_CPUS_PER_TASK / ${#WORKERS[@]}))
(( THREADS > 0 )) || THREADS=1
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"
LOCK="$(mktemp "${TMPDIR:-/tmp}/baseline_model_load.XXXXXX")"
export LEROBOT_MODEL_LOAD_LOCK="${LOCK}"
PIDS=()
ERROR_LOGS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  rm -f "${LOCK}"
}
trap cleanup EXIT
for row in "${WORKERS[@]}"; do
  IFS=$'\t' read -r tasks tag <<< "$row"
  log="${EVAL_WORK_DIR}/logs/${BASELINE}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${GROUP}_${tag}"
  echo "[${BASELINE}] group=${GROUP} tasks=${tasks}; logs=${log}.{out,err}"
  "${SIM_PYTHON}" "${SHARED_DIR}/run_eval.py" --tasks "${tasks}" --tag "${tag}" >"${log}.out" 2>"${log}.err" &
  PIDS+=("$!")
  ERROR_LOGS+=("${log}.err")
done
STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[index]}"; then
    :
  else
    STATUS=$?
    tail -45 "${ERROR_LOGS[index]}" >&2 || true
  fi
  # Do not signal a reaped PID later: the OS could reuse it during video merging.
  unset 'PIDS[index]'
done
# Merge even successful partial chunks if a peer failed. Never mask a rollout failure.
if ! compgen -G "${EVAL_OUT_DIR}/metrics/eval_info*.json" > /dev/null; then
  echo "No evaluation metrics were produced; skipping merge. See worker stderr above." >&2
  (( STATUS != 0 )) || STATUS=1
  exit "${STATUS}"
fi
PI05_SRC="${PROJECT_ROOT}/lerobot/examples/libero/configs/train_pi05/PT/eval/src"
WANDB_ARG=""; [[ "${WANDB_ENABLE}" != true ]] || WANDB_ARG="${WANDB_PROJECT}"
set +e
(
  set -e
  flock -x 9
  "${SIM_PYTHON}" "${PI05_SRC}/merge_eval_chunks.py" --out_dir="${EVAL_OUT_DIR}" \
    --expected_tasks="${EVAL_EXPECTED_TASKS}" --wandb_project="${WANDB_ARG}" --job_name="${WANDB_RUN_NAME}"
  if [[ "${VIDEO_ENABLE}" = true ]] && (( PANEL_COUNT > 1 )); then
    "${SIM_PYTHON}" "${PI05_SRC}/stitch_panels.py" --out_dir="${EVAL_OUT_DIR}" --grid_columns="${GRID_COLUMNS}"
  fi
) 9>"${EVAL_OUT_DIR}/.merge_eval_chunks.lock"
MERGE_STATUS=$?
set -e
(( STATUS == 0 )) || exit "${STATUS}"
exit "${MERGE_STATUS}"
