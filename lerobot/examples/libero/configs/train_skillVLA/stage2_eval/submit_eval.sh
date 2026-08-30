#!/usr/bin/env bash
# Submit Stage-2 evaluation with several independent evaluators per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE2_EVAL_CONFIG:-${SCRIPT_DIR}/stage2_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON=/usr/bin/python3
STAGE2_EVAL_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${STAGE2_EVAL_EXPORTS}"

# eval_num_gpus is a ceiling. Request only the GPUs needed by the task x panel
# grid and pack at most eval_max_workers_per_gpu evaluators onto each device.
PLANNED_GPUS="${EVAL_NUM_GPUS}"
[ -z "${SLURM_JOB_ID:-}" ] || PLANNED_GPUS=1
PACKING_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../eval_gpu_packing.py" \
    --items-json "${TASK_IDS}" \
    --panel-count "${MODEL_COUNT}" \
    --gpus "${PLANNED_GPUS}" \
    --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" \
    --shell
)"
eval "${PACKING_EXPORTS}"

# Keep the historical Stage-2 meaning of slurm.time: it is the budget for one
# task x panel unit. Workers on the same GPU run concurrently, so scale only by
# the largest sequential unit count owned by one logical worker.
JOB_TIME="$(
  "${BOOTSTRAP_PYTHON}" - "${EVAL_TIME}" "${EVAL_MAX_UNITS_PER_WORKER}" <<'PY'
import math, sys

spec, factor = sys.argv[1].strip(), int(sys.argv[2])
days = 0
if "-" in spec:
    day_part, spec = spec.split("-", 1)
    days = int(day_part)
parts = [int(value) for value in spec.split(":")]
if len(parts) == 1:
    hours, minutes, seconds = 0, parts[0], 0
elif len(parts) == 2:
    hours, minutes, seconds = 0, parts[0], parts[1]
else:
    hours, minutes, seconds = parts
seconds = (((days * 24 + hours) * 60 + minutes) * 60 + seconds) * factor
minutes = math.ceil(seconds / 60)
out_days, minutes = divmod(minutes, 24 * 60)
out_hours, out_minutes = divmod(minutes, 60)
print(f"{out_days}-{out_hours:02d}:{out_minutes:02d}:00")
PY
)"

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
EVAL_VENV_ARCHIVE=""
if [ "${EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive \
    "${PROJECT_ROOT}" "Stage-2 eval venv")"; then
    EVAL_VENV_ARCHIVE=""
    echo "Stage-2 eval: venv archive unavailable; using shared venv." >&2
  fi
fi
export EVAL_VENV_ARCHIVE

for artifact in "${POLICY_PATH}" "${FSQ_PATH}" "${SKILL_DATASET_DIR}"; do
  [ -e "${artifact}" ] || { echo "Missing artifact: ${artifact}" >&2; exit 1; }
done

SBATCH_ARGS=(
  --job-name="${STAGE2_EVAL_JOB_NAME:-S2eval}"
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${JOB_TIME}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit Stage-2 eval"
echo "  panels : ${MODEL_COUNT} (stage2/prior structure preserved)"
echo "  output : ${EVAL_OUT_DIR}"
echo "  GPUs   : ${EVAL_PHYSICAL_GPU_COUNT} physical (requested ${EVAL_NUM_GPUS})"
echo "  workers: ${EVAL_LOGICAL_WORKER_COUNT} total, max ${EVAL_MAX_WORKERS_PER_GPU}/GPU"
echo "  time   : ${EVAL_TIME} x ${EVAL_MAX_UNITS_PER_WORKER} max sequential units -> ${JOB_TIME}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : local worker group via srun in allocation ${SLURM_JOB_ID}"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_PHYSICAL_GPU_COUNT}" -le 1 ]; then
  echo "  mode   : one sbatch job"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_PHYSICAL_GPU_COUNT - 1))%${EVAL_PHYSICAL_GPU_COUNT}"
  echo "  mode   : Slurm array ${ARRAY_SPEC}"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
