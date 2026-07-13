#!/usr/bin/env bash
# Run SkillVLA-data evals off the FINAL artifacts in {run_dir}:
#   skillvla/  +  dino.npz  +  FSQ.pt  +  raw video ({dataset_root}/{source_dataset})
# Outputs: {run_dir}/eval/{dino,skillset,fsq_patch,fsq_recon}/
#
# Eval knobs/slurm come from eval_config.yaml; paths from train_skillVLA_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
BUILD_SRC_DIR="${SCRIPT_DIR}/../build_data/src"   # train_skillVLA_config.py lives with build_data
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/../build_data/train_skillVLA_config.yaml}"
EVAL_CONFIG_PATH="${EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
EVAL_CONFIG_PATH="$(snapshot_config "${EVAL_CONFIG_PATH}")"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${BUILD_SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${BUILD_SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/eval_config.py" --config "${EVAL_CONFIG_PATH}" --shell)"

for path in "${SKILLVLA_DATASET_DIR}" "${FSQ_COPY_PATH}" "${RAW_DATASET_DIR}"; do
  if [ ! -e "${path}" ]; then
    echo "Final artifact missing: ${path}" >&2
    echo "Run submit_build_skillvla.sh first (and keep cleanup off if you removed intermediates needed elsewhere)." >&2
    exit 1
  fi
done

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit SkillVLA eval"
echo "  skillvla : ${SKILLVLA_DATASET_DIR}"
echo "  FSQ.pt   : ${FSQ_COPY_PATH}"
echo "  out      : ${EVAL_DIR}"
echo "  run      : dino=${EVAL_RUN_DINO} skillset=${EVAL_RUN_SKILLSET} fsq_patch=${EVAL_RUN_FSQ_PATCH} fsq_recon=${EVAL_RUN_FSQ_RECON}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  echo "  mode     : srun (reusing allocation ${SLURM_JOB_ID})"
  TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" EVAL_CONFIG="${EVAL_CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" \
    srun "${SRC_DIR}/eval.sbatch"
else
  echo "  mode     : sbatch (new job)"
  TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" EVAL_CONFIG="${EVAL_CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
