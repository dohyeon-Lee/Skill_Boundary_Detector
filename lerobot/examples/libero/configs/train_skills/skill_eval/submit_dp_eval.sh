#!/usr/bin/env bash
# Inputs:
#   config : ./dp_eval_config.yaml  (roots + DP selection + HTML knobs + slurm)
# Outputs:
#   DP : ./outputs/dp_skillset/{dataset}/{dp_tag}_ck{ckpt}{suffix}.html
#
# Submit the DP skill-boundary eval (boxed start/end frames per skill + the
# multimodality curve). FSQ-independent; use submit_fsq_eval.sh for the FSQ eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}"

# This script runs exactly the DP eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_DP=true
export EVAL_RUN_FSQ=false

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

# Evaluation-only knobs + Slurm resources.
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/eval_config.py" --config "${EVAL_CONFIG}" --shell)"

# The selected artifact must already exist. Its manifest replaces all duplicated
# DP/checkpoint/dataset/threshold fields that used to live in this eval yaml.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
if [[ "${DP_EVAL_SKILLSET_DIR}" = /* ]]; then
  SELECTED_SKILLSET_DIR="${DP_EVAL_SKILLSET_DIR}"
else
  SELECTED_SKILLSET_DIR="${PROJECT_ROOT}/${DP_EVAL_SKILLSET_DIR}"
fi
DP_MANIFEST="${SELECTED_SKILLSET_DIR}/skillset_manifest.json"
if [ ! -f "${DP_MANIFEST}" ]; then
  echo "Skillset manifest not found: ${DP_MANIFEST}" >&2
  exit 1
fi
TARGET_DATASET="$("${BOOTSTRAP_PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["dataset_name"])' "${DP_MANIFEST}")"

# The shared resolver is retained only for global roots/runtime environment.
eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" \
  --config "${TRAIN_CONFIG}" --dataset "${TARGET_DATASET}" --shell)"

SBATCH_ARGS=(
  --job-name=dp_eval
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

echo "Submit DP skill-boundary eval"
echo "  skillset    : ${SELECTED_SKILLSET_DIR}"
echo "  dataset     : ${TARGET_DATASET} (from manifest)"
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
