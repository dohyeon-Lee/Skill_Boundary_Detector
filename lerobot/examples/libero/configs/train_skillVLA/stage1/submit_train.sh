#!/usr/bin/env bash
# Submit SkillVLA Stage-1 training (policy.type=skill_expert).
#   ./submit_train.sh [staged_1|staged_2|single]   (default: staged_1)
#   (login) resolve config + check the skillvla dataset → sbatch src/train_<experiment>.sbatch
#     staged_1 → VSA base (no Oracle).  staged_2 → freeze base + Oracle (warm-start from 1-1).
#     single   → one run, Oracle from scratch (CFG A/B dropout).
# The skillvla dataset comes from configs/train_skillVLA/build_data (run that first if missing).
set -euo pipefail

EXPERIMENT="${1:-staged_1}"   # staged_1 | staged_2 | single → submits src/train_${EXPERIMENT}.sbatch
case "${EXPERIMENT}" in
  staged_1|staged_2|single) ;;
  *) echo "usage: $0 [staged_1|staged_2|single]  (got '${EXPERIMENT}')" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage1
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SCRIPT_DIR}/stage1_train_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --shell)"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

case "${EXPERIMENT}" in
  staged_1) _OUT="${STAGED_1_1_DIR}" ;;
  staged_2) _OUT="${STAGED_1_2_DIR}" ;;
  single)   _OUT="${SINGLE_DIR}" ;;
esac
echo "Submit Stage-1 (skill_expert)  experiment=${EXPERIMENT}"
echo "  run      : ${BASE_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  init     : ${PI_BASE:-scratch}"
echo "  output   : ${_OUT}"
echo "  oracle   : r_dim=${ORACLE_R_DIM} kl=${ORACLE_KL_WEIGHT} dropout=${ORACLE_DROPOUT_P}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

STAGE1_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train_${EXPERIMENT}.sbatch"
