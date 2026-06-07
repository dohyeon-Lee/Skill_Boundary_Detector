#!/usr/bin/env bash
# Submit SkillVLA Stage-2 training (policy.type=skill_vla).
#   (login) resolve config + check artifacts → sbatch train.sbatch
# Prereqs: the skillvla dataset (configs/train_skillVLA/build_data) and a Stage-1 skill_expert
# checkpoint (configs/train_skillVLA/stage1, named by stage1_run_name in the yaml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage2
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE2_TRAIN_CONFIG:-${SCRIPT_DIR}/stage2_train_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_train_config.py" --config "${CONFIG_PATH}" --shell)"

# ── prerequisite artifacts ──
# skillvla dataset ← configs/train_skillVLA/build_data ; Stage-1 checkpoint ← configs/train_skillVLA/stage1.
# FSQ.pt is eval-only (terminator) so it is NOT required to train.
if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
  exit 1
fi
if [ ! -e "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-1 checkpoint: ${STAGE1_CHECKPOINT_PATH}" >&2
  echo "Train Stage-1 first: configs/train_skillVLA/stage1/submit_train.sh (set stage1_run_name in the yaml)" >&2
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

echo "Submit SkillVLA Stage-2"
echo "  run      : ${PT_RUN_NAME}  (arch=${EXPERT_ARCH})"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  expert   : ${STAGE1_CHECKPOINT_PATH}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

STAGE2_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
