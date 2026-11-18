#!/usr/bin/env bash
# Shared launcher. Model-specific PT/eval/submit_eval.sh supplies family/config/workdir.
set -euo pipefail
REQUESTED_BASELINE="${1:?family}"; WORK_DIR="${2:?eval directory}"; CONFIG_PATH="${3:?yaml}"
shift 3
SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SHARED_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${CONFIGS_DIR}/../../../.." && pwd)"
BOOTSTRAP="${REPO_DIR}/.venv/bin/python"
[[ -x "${BOOTSTRAP}" ]] || BOOTSTRAP=python3
DRY_RUN="${BASELINE_EVAL_DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in --dry-run) DRY_RUN=1;; *) echo "Unsupported argument: $arg" >&2; exit 2;; esac
done
source "${CONFIGS_DIR}/src/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
eval "$("${BOOTSTRAP}" "${SHARED_DIR}/config.py" --config "${CONFIG_PATH}" --work-dir "${WORK_DIR}" --shell)"
[[ "${BASELINE}" = "${REQUESTED_BASELINE}" ]] || { echo "Wrong baseline in YAML: ${BASELINE}" >&2; exit 1; }
PLANNED_GPUS="${EVAL_NUM_GPUS}"
[[ -z "${SLURM_JOB_ID:-}" ]] || PLANNED_GPUS=1
eval "$("${BOOTSTRAP}" "${CONFIGS_DIR}/src/eval_gpu_packing.py" --items-json "${TASK_IDS}" \
  --gpus "${PLANNED_GPUS}" --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" --shell)"
echo "Submit ${BASELINE} closed-loop evaluation"
echo "  panels: ${PANEL_SUMMARY}"
echo "  tasks: ${TASK_IDS}, ${N_EPISODES} episodes/task; replay ${N_ACTION_STEPS} actions"
echo "  GPUs: ${EVAL_PHYSICAL_GPU_COUNT}; logical workers: ${EVAL_LOGICAL_WORKER_COUNT}"
echo "  model environment: ${MODEL_PYTHON}"
echo "  output: ${EVAL_OUT_DIR} (resume=${EVAL_RESUME})"
if [[ "${DRY_RUN}" = 1 ]]; then echo "Dry run: no jobs submitted."; exit 0; fi
mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"
ARGS=(--job-name="${BASELINE}eval" --partition="${EVAL_PARTITION}" --qos="${EVAL_QOS}" \
  --gres="${EVAL_GRES}" --cpus-per-task="${EVAL_CPUS_PER_TASK}" --mem="${EVAL_MEM}" --time="${EVAL_TIME}")
[[ -z "${EVAL_NODELIST}" ]] || ARGS+=(--nodelist="${EVAL_NODELIST}")
[[ -z "${EVAL_EXCLUDE_NODES}" ]] || ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
source "${CONFIGS_DIR}/src/submit_job.sh"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun bash "${SHARED_DIR}/eval.sbatch"
elif [[ "$(submit_job_scheduler)" = local ]]; then
  for ((group=0; group<EVAL_PHYSICAL_GPU_COUNT; group++)); do
    BASELINE_GPU_GROUP="${group}" submit_job "${ARGS[@]}" "${SHARED_DIR}/eval.sbatch"
  done
elif (( EVAL_PHYSICAL_GPU_COUNT > 1 )); then
  ARRAY_SPEC="${BASELINE_EVAL_ARRAY:-0-$((EVAL_PHYSICAL_GPU_COUNT-1))%${EVAL_PHYSICAL_GPU_COUNT}}"
  echo "  Slurm array groups: ${ARRAY_SPEC}"
  submit_job --array="${ARRAY_SPEC}" \
    --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err "${ARGS[@]}" "${SHARED_DIR}/eval.sbatch"
else
  submit_job "${ARGS[@]}" "${SHARED_DIR}/eval.sbatch"
fi
