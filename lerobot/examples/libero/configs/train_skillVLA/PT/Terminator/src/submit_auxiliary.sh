#!/usr/bin/env bash
# Submit unified PT/FT skill predictor + FSQ terminator training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${SKILL_AUX_TRAIN_CONFIG:-${TERMINATOR_TRAIN_CONFIG:-${SCRIPT_DIR}/../terminator_train_config.yaml}}"
if [ "$#" -ne 0 ]; then
  echo "Usage: $0 (select the component YAML with SKILL_AUX_TRAIN_CONFIG)" >&2
  exit 2
fi

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/src/snapshot_config.sh" ]; do
  CONFIG_LIB="$(dirname "${CONFIG_LIB}")"
done
source "${CONFIG_LIB}/src/snapshot_config.sh"
source "${CONFIG_LIB}/src/submit_job.sh"   # sbatch, or a local run where the server has no Slurm
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${CONFIG_LIB}/../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
if [ -n "${TERMINATOR_ARCHITECTURE_OVERRIDE:-}" ]; then
  CONFIG_ARGS+=(--architecture "${TERMINATOR_ARCHITECTURE_OVERRIDE}")
fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_train_config.py" "${CONFIG_ARGS[@]}"
)"; then
  echo "Auxiliary configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"
: "${SKILLVLA_DATASET_DIR:?Bootstrap did not export SKILLVLA_DATASET_DIR}"
if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing SkillVLA dataset: ${SKILLVLA_DATASET_DIR}" >&2
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

cd "${SKILL_AUX_SUBMIT_DIR:-${SCRIPT_DIR}/..}"
mkdir -p logs
echo "Submit auxiliary-only training"
echo "  init    : ${INITIALIZATION_MODE}"
echo "  targets : ${TRAINING_MODE}"
echo "  run     : ${RUN_NAME}"
echo "  dataset : ${SKILLVLA_DATASET_DIR}"
echo "  output  : ${OUTPUT_DIR}"
if (( PREDICTOR_PARALLEL_COUNT > 1 )); then
  echo "  parallel: ${PREDICTOR_PARALLEL_COUNT} independent Predictors on the SAME GPU"
  "${BOOTSTRAP_PYTHON}" -c '
import json, os
for run in json.loads(os.environ["PREDICTOR_PARALLEL_RUNS"]):
    print("    {mode}: {output_dir}".format(**run))
'
  echo "  resources: batch_size=${BATCH_SIZE} and workers=${NUM_WORKERS} PER MODEL; Slurm CPU/RAM are job totals"
fi
if (( TERMINATOR_PARALLEL_COUNT > 1 )); then
  echo "  parallel: ${TERMINATOR_PARALLEL_COUNT} independent Terminators on the SAME GPU"
  "${BOOTSTRAP_PYTHON}" -c '
import json, os
for run in json.loads(os.environ["TERMINATOR_PARALLEL_RUNS"]):
    print("    {architecture}: {output_dir}".format(**run))
'
  echo "  resources: batch_size=${BATCH_SIZE} and workers=${NUM_WORKERS} PER MODEL; Slurm CPU/RAM are job totals"
fi

if [ "${SKILL_AUX_DRY_RUN:-0}" = 1 ]; then
  printf 'Dry run: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
  printf '\n'
  exit 0
fi

SKILL_AUX_TRAIN_CONFIG="${CONFIG_PATH}" \
SKILL_AUX_SRC_DIR="${SRC_DIR}" \
TERMINATOR_ARCHITECTURE_OVERRIDE="${TERMINATOR_ARCHITECTURE_OVERRIDE:-}" \
  submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
