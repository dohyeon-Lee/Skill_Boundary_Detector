#!/usr/bin/env bash
# Submit SkillVLA Stage-1 training (policy.type=skill_expert).
#   ./submit_train.sh [joint|staged]   (default: joint)
#   (login) resolve config + check the skillvla dataset → sbatch train_<mode>.sbatch
#     joint  → one run.   staged → two runs ({run}/1-1 then {run}/1-2).
# The skillvla dataset comes from configs/train_skillVLA/build_data (run that first if missing).
set -euo pipefail

EXPERIMENT="${1:-joint_plain}"   # joint_plain | joint_gated | staged_1 | staged_2 → submits src/train_${EXPERIMENT}.sbatch
case "${EXPERIMENT}" in
  joint_plain)  EMIT_MODE=joint;  EMIT_LOSS=plain ;;
  joint_gated)  EMIT_MODE=joint;  EMIT_LOSS=weighted_gated ;;
  staged_1|staged_2) EMIT_MODE=staged; EMIT_LOSS=plain ;;
  *) echo "usage: $0 [joint_plain|joint_gated|staged_1|staged_2]  (got '${EXPERIMENT}')" >&2; exit 1 ;;
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

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --mode "${EMIT_MODE}" --loss-mode "${EMIT_LOSS}" --shell)"

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

echo "Submit Stage-1 (skill_expert)  experiment=${EXPERIMENT}"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  init     : ${PI_BASE:-scratch}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

STAGE1_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train_${EXPERIMENT}.sbatch"
