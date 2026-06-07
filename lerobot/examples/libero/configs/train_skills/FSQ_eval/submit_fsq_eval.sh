#!/usr/bin/env bash
# Inputs:
#   roots        : ../train_skills_config.yaml  (skills / DINO / dataset / FSQ_outputs root)
#   eval knobs   : ./fsq_eval_config.yaml       (run_name, checkpoint, N_ACTION_STEPS, samples, slurm)
#   FSQ model    : {project_root}/FSQ_outputs/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch*.pt)
# Outputs:
#   latents+HTML : ./outputs/{fsq_eval_run_name}/{epoch}/{skill_latents.npz, fsq_eval.html}
#
# Submit unified FSQ evaluation (encoder + decoder) using Slurm values from fsq_eval_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../train_skills_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_eval_config.yaml}"

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

# Root paths from the shared train_skills config.
if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi
# Evaluation-only knobs + slurm.
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/fsq_eval_config.py" --config "${EVAL_CONFIG}" --shell)"

SBATCH_ARGS=(
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
echo "  run         : ${FSQ_EVAL_RUN_NAME}"
echo "  checkpoint  : ${FSQ_EVAL_CHECKPOINT}"
echo "  skills      : ${SKILLSET_DIR}/skills"
echo "  dataset     : ${DATASET_ROOT}/${TARGET_DATASET}"
echo "  FSQ outputs : $(dirname "${FSQ_OUTPUT_DIR}")/${FSQ_EVAL_RUN_NAME}"
echo "  slurm       : partition=${FSQ_EVAL_PARTITION} qos=${FSQ_EVAL_QOS} gres=${FSQ_EVAL_GRES}"

TRAIN_SKILLS_CONFIG="${TRAIN_CONFIG}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${EVAL_SRC_DIR}/fsq_eval.sbatch"
