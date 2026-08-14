#!/usr/bin/env bash
# Submit BSQ training (FSQ-original structure, Binary Spherical Quantization).
# Outputs (same outputs/FSQ root; run name carries a bsq{code_dim} tag):
#   run          : {project_root}/outputs/FSQ/{fsq_run_name}
#   checkpoints  : {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BSQ_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/bsq_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TARGET_DATASET}" ]; then
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${BSQ_SRC_DIR}/bsq_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${BSQ_SRC_DIR}/bsq_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "${RESOLVED_SETTINGS}"

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run build_data/submit_build_data.sh first." >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${FSQ_TRAIN_PARTITION}"
  --qos="${FSQ_TRAIN_QOS}"
  --gres="${FSQ_TRAIN_GRES}"
  --cpus-per-task="${FSQ_TRAIN_CPUS_PER_TASK}"
  --mem="${FSQ_TRAIN_MEM}"
  --time="${FSQ_TRAIN_TIME}"
)

if [ -n "${FSQ_TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_TRAIN_NODELIST}")
fi
if [ -n "${FSQ_TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs_train

echo "Submit BSQ train"
echo "  dataset     : ${TARGET_DATASET}"
echo "  skillset    : ${SKILLSET_DIR}/skills"
echo "  quantizer   : bsq code_dim=${BSQ_CODE_DIM} (codebook $((2 ** BSQ_CODE_DIM)))"
echo "  output      : ${FSQ_OUTPUT_DIR}"
echo "  slurm       : partition=${FSQ_TRAIN_PARTITION} qos=${FSQ_TRAIN_QOS} gres=${FSQ_TRAIN_GRES}"

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${BSQ_SRC_DIR}/train_bsq.sbatch"
