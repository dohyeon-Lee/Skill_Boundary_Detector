#!/usr/bin/env bash
# Inputs:
#   config    : ./fsq_eval_config.yaml  (FSQ run folder + checkpoint list)
#   FSQ model : {outputs_root}/FSQ/{fsq_eval_run_name}/FSQ_epoch*.pt
# Outputs:
#   FSQ : ./outputs/{fsq_eval_run_name}/{epoch}/fsq_eval.html per checkpoint
#
# Submit the FSQ reconstruction/termination/progress eval. The FSQ reads its OWN
# skillset (recorded in fsq_meta.json), which already exists since the FSQ trained
# on it. Both eval launchers require their selected artifacts to exist up front.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_eval_config.yaml}"

# This script runs exactly the FSQ eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_FSQ=true
export EVAL_RUN_DP=false

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${EVAL_CONFIG}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
EVAL_CONFIG="$(snapshot_config "${EVAL_CONFIG}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Resolve and validate every checkpoint before reserving GPUs.
if ! RESOLVED_SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/eval_config.py" \
    --config "${EVAL_CONFIG}" --shell
)"; then
  echo "FSQ evaluation bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED_SETTINGS}"
read -r -a FSQ_CHECKPOINTS <<< "${FSQ_EVAL_CHECKPOINTS}"
FSQ_CHECKPOINT_COUNT="${#FSQ_CHECKPOINTS[@]}"

SBATCH_ARGS=(
  --job-name=fsq_eval
  --partition="${FSQ_EVAL_PARTITION}"
  --qos="${FSQ_EVAL_QOS}"
  --gres="${FSQ_EVAL_GRES}"
  --cpus-per-task="${FSQ_EVAL_CPUS_PER_TASK}"
  --mem="${FSQ_EVAL_MEM}"
  --time="${FSQ_EVAL_TIME}"
)
if [ -n "${FSQ_EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_EVAL_NODELIST}")
fi
if [ -n "${FSQ_EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs outputs

echo "Submit FSQ eval"
echo "  checkpoints : ${FSQ_EVAL_CHECKPOINTS}"
echo "  parallel GPU: ${FSQ_EVAL_NUM_GPUS} (one GPU per checkpoint)"
echo "  dataset     : ${FSQ_EVAL_DATASET_DIR}"
echo "  skillset    : ${FSQ_EVAL_SKILLSET_DIR}"
echo "  slurm       : partition=${FSQ_EVAL_PARTITION} qos=${FSQ_EVAL_QOS} gres=${FSQ_EVAL_GRES}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # One exclusive one-GPU step per checkpoint. Slurm starts as many steps as
  # the existing allocation can serve concurrently.
  echo "  mode        : parallel srun steps (allocation ${SLURM_JOB_ID})"
  PIDS=()
  STATUS=0
  for INDEX in "${!FSQ_CHECKPOINTS[@]}"; do
    CHECKPOINT="${FSQ_CHECKPOINTS[${INDEX}]}"
    FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
      FSQ_EVAL_CHECKPOINT_OVERRIDE="${CHECKPOINT}" \
      srun --exclusive --nodes=1 --ntasks=1 --gres="${FSQ_EVAL_GRES}" \
        "${EVAL_SRC_DIR}/eval.sbatch" &
    PIDS+=("$!")
    if [ "${#PIDS[@]}" -eq "${FSQ_EVAL_NUM_GPUS}" ] || \
       [ "${INDEX}" -eq "$((FSQ_CHECKPOINT_COUNT - 1))" ]; then
      for PID in "${PIDS[@]}"; do
        wait "${PID}" || STATUS=1
      done
      PIDS=()
    fi
  done
  exit "${STATUS}"
elif [ "${FSQ_CHECKPOINT_COUNT}" -gt 1 ]; then
  FSQ_EVAL_FANOUT="${FSQ_EVAL_CHECKPOINTS}"
  export FSQ_EVAL_FANOUT
  ARRAY_SPEC="0-$((FSQ_CHECKPOINT_COUNT - 1))%${FSQ_EVAL_NUM_GPUS}"
  echo "  mode        : Slurm array ${ARRAY_SPEC}"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
    sbatch --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${EVAL_SRC_DIR}/eval.sbatch"
else
  echo "  mode        : sbatch (new job)"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
    sbatch "${SBATCH_ARGS[@]}" "${EVAL_SRC_DIR}/eval.sbatch"
fi
