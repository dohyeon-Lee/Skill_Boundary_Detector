#!/usr/bin/env bash
# Inputs:
#   config    : ./fsq_eval_config.yaml  (roots + FSQ selection + HTML knobs + slurm)
#   FSQ model : {project_root}/outputs/FSQ/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch*.pt)
# Outputs:
#   FSQ : ./outputs/{fsq_eval_run_name}/{epoch}/fsq_eval.html
#
# Submit the FSQ reconstruction/termination/progress eval. The FSQ reads its OWN
# skillset (recorded in fsq_meta.json), which already exists since the FSQ trained
# on it — so no build_data auto-chaining is needed here (unlike submit_dp_eval.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/fsq_eval_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_eval_config.yaml}"

# This script runs exactly the FSQ eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_FSQ=true
export EVAL_RUN_DP=false

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${TRAIN_CONFIG}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
TRAIN_CONFIG="$(snapshot_config "${TRAIN_CONFIG}")"
EVAL_CONFIG="$(snapshot_config "${EVAL_CONFIG}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Evaluation-only knobs + slurm.
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/eval_config.py" --config "${EVAL_CONFIG}" --shell)"
# Shared resolver (roots; the fallback SKILLSET_DIR — fsq_meta.json normally overrides it in the job).
if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi

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
echo "  FSQ run     : ${FSQ_EVAL_RUN_NAME} (ckpt ${FSQ_EVAL_CHECKPOINT})"
echo "  dataset     : ${DATASET_ROOT}/${TARGET_DATASET}"
echo "  slurm       : partition=${FSQ_EVAL_PARTITION} qos=${FSQ_EVAL_QOS} gres=${FSQ_EVAL_GRES}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  echo "  mode        : srun (reusing allocation ${SLURM_JOB_ID})"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" TRAIN_SKILLS_CONFIG="${TRAIN_CONFIG}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" TRAIN_DATA="${TARGET_DATASET}" \
    srun "${EVAL_SRC_DIR}/eval.sbatch"
else
  echo "  mode        : sbatch (new job)"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" TRAIN_SKILLS_CONFIG="${TRAIN_CONFIG}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" TRAIN_DATA="${TARGET_DATASET}" \
    sbatch "${SBATCH_ARGS[@]}" "${EVAL_SRC_DIR}/eval.sbatch"
fi
