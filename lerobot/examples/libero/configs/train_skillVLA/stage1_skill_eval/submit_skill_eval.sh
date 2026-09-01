#!/usr/bin/env bash
# Submit policy x episode workers for the multi-policy skill evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_SKILL_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_skill_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

# Config resolution and worker planning need only the standard library.  Do not
# cold-import the shared project venv on the submit path.
BOOTSTRAP_PYTHON=/usr/bin/python3
STAGE1_SKILL_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_skill_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${STAGE1_SKILL_EVAL_EXPORTS}"

PLANNED_GPUS="${EVAL_NUM_GPUS}"
[ -z "${SLURM_JOB_ID:-}" ] || PLANNED_GPUS=1
PACKING_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../eval_gpu_packing.py" \
    --unit-count "${EVAL_WORK_UNIT_COUNT}" \
    --gpus "${PLANNED_GPUS}" \
    --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" \
    --shell
)"
eval "${PACKING_EXPORTS}"
export SKILL_EVAL_WORKER_COUNT="${EVAL_LOGICAL_WORKER_COUNT}"

# Turn many random Lustre reads during Python imports into one sequential copy
# per allocated node. All local workers then share the staged environment.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
EVAL_VENV_ARCHIVE=""
if [ "${EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive \
    "${PROJECT_ROOT}" "Stage-1 skill eval venv")"; then
    EVAL_VENV_ARCHIVE=""
    echo "Stage-1 skill eval: venv archive unavailable; using shared venv." >&2
  fi
fi
export EVAL_VENV_ARCHIVE

SBATCH_ARGS=(
  --job-name="${STAGE1_SKILL_EVAL_JOB_NAME:-S1skill}"
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit Stage-1 multi-policy skill eval"
echo "  policies : ${MODEL_COUNT} (${ARCHITECTURE_LABEL})"
echo "  MAIN     : ${MAIN_TERMINATOR_LABEL} (${MAIN_TERMINATOR_VARIANT})"
echo "  display  : ${TERMINATOR_MODEL_LABEL} (${TERMINATOR_MODEL_VARIANT})"
echo "  display rule: ${TERMINATOR_MODEL_END_MODE} term=${TERMINATOR_MODEL_END_THRESHOLD} progress=${TERMINATOR_MODEL_PROGRESS_THRESHOLD}"
echo "  tasks    : ${TARGET_TASK} dataset=${DATASET_TASK_IDS} env=${TASK_IDS}"
echo "  episodes : ${EPISODES_PER_TASK}/task (${EPISODE_SELECTION})"
echo "  shift    : ±${TIME_SHIFT_OFFSET}"
echo "  GPUs     : ${EVAL_PHYSICAL_GPU_COUNT} physical (requested ${EVAL_NUM_GPUS})"
echo "  workers  : ${EVAL_LOGICAL_WORKER_COUNT} total, max ${EVAL_MAX_WORKERS_PER_GPU}/GPU"
echo "  output   : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode     : local worker group via srun in allocation ${SLURM_JOB_ID}"
  STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_PHYSICAL_GPU_COUNT}" -le 1 ]; then
  echo "  mode     : one sbatch job"
  STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_PHYSICAL_GPU_COUNT - 1))%${EVAL_PHYSICAL_GPU_COUNT}"
  echo "  mode     : Slurm array ${ARRAY_SPEC}"
  STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}" STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
