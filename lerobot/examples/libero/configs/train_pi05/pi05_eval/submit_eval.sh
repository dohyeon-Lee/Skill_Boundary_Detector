#!/usr/bin/env bash
# Submit pi05 evaluation. Task ids are split into logical workers that are packed onto at most
# eval_num_gpus one-GPU Slurm array elements (eval_max_workers_per_gpu workers per GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${PI05_EVAL_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
# One frozen resolution, exported to every array element (the job never re-reads the yaml).
PI05_EVAL_EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/pi05_eval_config.py" --config "${CONFIG_PATH}" --shell)"
eval "${PI05_EVAL_EXPORTS}"

PLANNED_GPUS="${EVAL_NUM_GPUS}"
[ -z "${SLURM_JOB_ID:-}" ] || PLANNED_GPUS=1
PACKING_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../../train_skillVLA/eval_gpu_packing.py" \
    --items-json "${TASK_IDS}" \
    --gpus "${PLANNED_GPUS}" \
    --max-workers-per-gpu "${EVAL_MAX_WORKERS_PER_GPU}" \
    --shell
)"
eval "${PACKING_EXPORTS}"

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/node_local_venv.sh"
EVAL_VENV_ARCHIVE=""
if [ "${EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive "${PROJECT_ROOT}" "pi05 eval venv")"; then
    EVAL_VENV_ARCHIVE=""
    echo "pi05 eval: venv archive unavailable; using shared venv." >&2
  fi
fi
export EVAL_VENV_ARCHIVE

SBATCH_ARGS=(
  --job-name="${PI05_EVAL_JOB_NAME:-pi05eval}"
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
echo "Submit pi05 eval"
echo "  panels : ${PANEL_SUMMARY}"
echo "  tasks  : ${TARGET_TASK} task_ids=${TASK_IDS} episodes=${N_EPISODES}"
echo "  init   : $([ "${EPISODE_EXACT}" = true ] && echo "episode-exact ${EVAL_INIT_STATES_PATH}" || echo "benchmark init states, offset=${EPISODE_OFFSET}")"
echo "  video  : enable=${VIDEO_ENABLE} max_per_task=${MAX_VIDEOS_PER_TASK} wrist_panel=${VIDEO_SHOW_WRIST}"
echo "  output : ${EVAL_OUT_DIR} (resume=${EVAL_RESUME})"
echo "  GPUs   : ${EVAL_PHYSICAL_GPU_COUNT} physical (requested ${EVAL_NUM_GPUS})"
echo "  workers: ${EVAL_LOGICAL_WORKER_COUNT} total, max ${EVAL_MAX_WORKERS_PER_GPU}/GPU"

if [ "${PI05_EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: nothing submitted."
  exit 0
fi

if [ "${EVAL_RESUME}" != true ]; then
  # A fresh run must not merge chunk summaries left by a previous split of this output folder.
  rm -f "${EVAL_OUT_DIR}"/metrics/eval_info*.json "${EVAL_OUT_DIR}"/metrics/.wandb_logged 2>/dev/null || true
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun in allocation ${SLURM_JOB_ID}"
  PI05_EVAL_DIR="${SCRIPT_DIR}" srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_PHYSICAL_GPU_COUNT}" -le 1 ]; then
  echo "  mode   : one sbatch job"
  PI05_EVAL_DIR="${SCRIPT_DIR}" sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  ARRAY_SPEC="0-$((EVAL_PHYSICAL_GPU_COUNT - 1))%${EVAL_PHYSICAL_GPU_COUNT}"
  echo "  mode   : array ${ARRAY_SPEC}"
  PI05_EVAL_DIR="${SCRIPT_DIR}" sbatch --array="${ARRAY_SPEC}" \
    --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
    "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
