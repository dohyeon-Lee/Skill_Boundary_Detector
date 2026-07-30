#!/usr/bin/env bash
# Repair only the ext video portion of an already-built LangGap dataset.
# Usage:
#   ./submit_repair_ext_orientation.sh
#   DATASET=langgap_56_full_full OUTPUT_NAME=langgap_56_full_full_fixed ./submit_repair_ext_orientation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/langgap_dataset_config.py" --shell)"

DATASET="${DATASET:-langgap_56_full_full}"
OUTPUT_NAME="${OUTPUT_NAME:-${DATASET}_canonical_orientation}"
WORKERS="${WORKERS:-4}"
ENCODER_THREADS="${ENCODER_THREADS:-8}"

mkdir -p "${SCRIPT_DIR}/logs"
SBATCH_ARGS=(
  --job-name=repair_lg
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres="${CONVERT_GRES}"
  --cpus-per-task="${CONVERT_CPUS_PER_TASK}"
  --mem="${CONVERT_MEM}"
  --time="${CONVERT_TIME}"
  --output="${SCRIPT_DIR}/logs/%x_%j.out"
  --error="${SCRIPT_DIR}/logs/%x_%j.err"
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

COMMAND=(
  "${PROJECT_ROOT}/.venv/bin/python"
  "${SCRIPT_DIR}/repair_ext_orientation.py"
  --dataset "${DATASET}"
  --output-name "${OUTPUT_NAME}"
  --workers "${WORKERS}"
  --encoder-threads "${ENCODER_THREADS}"
)

printf -v WRAP '%q ' "${COMMAND[@]}"
echo "source : ${LANGGAP_ROOT}/${DATASET}"
echo "output : ${LANGGAP_ROOT}/${OUTPUT_NAME}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
