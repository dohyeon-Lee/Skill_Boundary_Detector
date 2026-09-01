#!/usr/bin/env bash
# Submit skill-start policy-noise trajectory overlay evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_SKILL_NOISE_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_skill_noise_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON=/usr/bin/python3
STAGE1_SKILL_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_skill_noise_eval_config.py" \
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

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
EVAL_VENV_ARCHIVE=""
if [ "${EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive \
    "${PROJECT_ROOT}" "Stage-1 skill noise eval venv")"; then
    EVAL_VENV_ARCHIVE=""
    echo "Stage-1 skill noise eval: venv archive unavailable; using shared venv." >&2
  fi
fi
export EVAL_VENV_ARCHIVE

SBATCH_ARGS=(
  --job-name="${STAGE1_SKILL_NOISE_EVAL_JOB_NAME:-S1noise}"
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
echo "Submit Stage-1 skill-start noise trajectory evaluation"
echo "  policies : ${MODEL_COUNT} (${ARCHITECTURE_LABEL})"
echo "  tasks    : ${TARGET_TASK} dataset=${DATASET_TASK_IDS} env=${TASK_IDS}"
echo "  envs     : ${ENVS_PER_TASK}/task (${EPISODE_SELECTION})"
echo "  noise    : ${NOISE_ROLLOUTS_PER_ENV} rollouts/environment"
echo "  code probe: ${NEIGHBOR_CODE_PROBE} (off | neighbor | neighbor_and_opposite | all)"
echo "  units    : ${EVAL_WORK_UNIT_COUNT} policy x environment x noise"
echo "  GPUs     : ${EVAL_PHYSICAL_GPU_COUNT} physical (requested ${EVAL_NUM_GPUS})"
echo "  workers  : ${EVAL_LOGICAL_WORKER_COUNT} total, max ${EVAL_MAX_WORKERS_PER_GPU}/GPU"
echo "  output   : ${EVAL_OUT_DIR}"

COMMON_ENV=(
  STAGE1_SKILL_EVAL_DIR="${SCRIPT_DIR}"
  STAGE1_SKILL_EVAL_CONFIG="${CONFIG_PATH}"
  STAGE1_SKILL_EVAL_MODE="noise_overlay"
  STAGE1_SKILL_EVAL_RUNNER="run_skill_noise_eval.py"
)
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode     : local worker group via srun in allocation ${SLURM_JOB_ID}"
  env "${COMMON_ENV[@]}" srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_PHYSICAL_GPU_COUNT}" -le 1 ]; then
  echo "  mode     : one sbatch job"
  env "${COMMON_ENV[@]}" sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_PHYSICAL_GPU_COUNT - 1))%${EVAL_PHYSICAL_GPU_COUNT}"
  echo "  mode     : Slurm array ${ARRAY_SPEC}"
  env "${COMMON_ENV[@]}" sbatch --array="${ARRAY_SPEC}" \
    --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
    "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
