#!/usr/bin/env bash
# Submit SkillVLA stage-3 PRE-TRAINING (PT).
#   (login) resolve config + check artifacts → sbatch train.sbatch
# Artifacts (skillvla dataset / FSQ.pt / dino.npz) come from the build_data pipeline
# (configs/train_skillVLA/build_data). Run that first if they are missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # skillVLA
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${SKILLVLA_TRAIN_CONFIG:-${SCRIPT_DIR}/skillVLA_train_config.yaml}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/skillVLA_train_config.py" --config "${CONFIG_PATH}" --shell)"

# ── prerequisite artifacts (built by configs/train_skillVLA/build_data) ──
for p in "${SKILLVLA_DATASET_DIR}" "${FSQ_CKPT}" "${DINO_TOKENS_PATH}"; do
  if [ ! -e "${p}" ]; then
    echo "Missing artifact: ${p}" >&2
    echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
    exit 1
  fi
done

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

echo "Submit SkillVLA PT"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  FSQ      : ${FSQ_CKPT}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

SKILLVLA_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
