#!/usr/bin/env bash
# Submit the standalone FSQ-terminator training.
#   (login) resolve config (SLURM args + paths) → check FSQ.pt/dino.npz/skillvla → sbatch train.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # terminator_train
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TERMINATOR_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_train_config.yaml}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_train_config.py" --config "${CONFIG_PATH}" --shell)"

for f in "${FSQ_PATH}" "${DINO_PATH}" "${SKILLVLA_DATASET_DIR}"; do
  if [ ! -e "${f}" ]; then echo "Missing: ${f}  (build the skillvla dataset + FSQ first)" >&2; exit 1; fi
done

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}"); fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}"); fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit FSQ-terminator training"
echo "  FSQ    : ${FSQ_PATH}"
echo "  data   : ${SKILLVLA_DATASET_DIR}"
echo "  slurm  : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

TERMINATOR_TRAIN_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
