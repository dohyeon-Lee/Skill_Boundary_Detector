#!/usr/bin/env bash
# Submit Stage-1 evaluation; eval_num_gpus splits task ids over a Slurm array.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON=/usr/bin/python3
STAGE1_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${STAGE1_EVAL_EXPORTS}"

PLANNED_GPUS="${EVAL_NUM_GPUS}"
[ -z "${SLURM_JOB_ID:-}" ] || PLANNED_GPUS=1
PACKING_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../eval_gpu_packing.py" \
    --items-json "${TASK_IDS}" \
    --gpus "${PLANNED_GPUS}" \
    --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" \
    --shell
)"
eval "${PACKING_EXPORTS}"

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
EVAL_VENV_ARCHIVE=""
if [ "${EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive \
    "${PROJECT_ROOT}" "Stage-1 eval venv")"; then
    EVAL_VENV_ARCHIVE=""
    echo "Stage-1 eval: venv archive unavailable; using shared venv." >&2
  fi
fi
export EVAL_VENV_ARCHIVE

for artifact in "${POLICY_PATH}" "${FSQ_PATH}" "${SKILL_DATASET_DIR}"; do
  [ -e "${artifact}" ] || { echo "Missing artifact: ${artifact}" >&2; exit 1; }
done

SBATCH_ARGS=(
  --job-name="${STAGE1_EVAL_JOB_NAME:-S1eval}"
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
echo "Submit Stage-1 eval"
echo "  models : ${MODEL_ARCHITECTURES}"
echo "  tasks  : ${TARGET_TASK} dataset=${DATASET_TASK_IDS} env=${TASK_IDS}"
echo "  output : ${EVAL_OUT_DIR}"
echo "  GPUs   : ${EVAL_PHYSICAL_GPU_COUNT} physical (requested ${EVAL_NUM_GPUS})"
echo "  workers: ${EVAL_LOGICAL_WORKER_COUNT} total, max ${EVAL_MAX_WORKERS_PER_GPU}/GPU"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun in allocation ${SLURM_JOB_ID}"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_PHYSICAL_GPU_COUNT}" -le 1 ]; then
  echo "  mode   : one sbatch job"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_PHYSICAL_GPU_COUNT - 1))%${EVAL_PHYSICAL_GPU_COUNT}"
  echo "  mode   : array ${ARRAY_SPEC}"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
