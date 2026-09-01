#!/usr/bin/env bash
# Submit the YAML-configured oracle init-state matcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ORACLE_MATCHING_CONFIG:-${SCRIPT_DIR}/oracle_matching_config.yaml}"
if [ "${1:-}" = "--config" ]; then
  [ $# -ge 2 ] || { echo "--config requires a YAML path" >&2; exit 2; }
  CONFIG_PATH="$2"
  shift 2
elif [ $# -gt 0 ]; then
  CONFIG_PATH="$1"
  shift
fi
[ $# -eq 0 ] || { echo "Unexpected arguments: $*" >&2; exit 2; }

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do
  PARENT="$(dirname "${CONFIG_LIB}")"
  [ "${PARENT}" != "${CONFIG_LIB}" ] || {
    echo "snapshot_config.sh not found above ${CONFIG_PATH}" >&2
    exit 1
  }
  CONFIG_LIB="${PARENT}"
done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

PROJECT_HINT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
BOOTSTRAP_PYTHON="${PROJECT_HINT}/.venv/bin/python"
ORACLE_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/oracle_matching_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${ORACLE_EXPORTS}"

if [ -f "${ORACLE_OUTPUT_PATH}" ] && [ "${ORACLE_OVERWRITE}" != true ]; then
  echo "Oracle init-state map already exists; nothing submitted: ${ORACLE_OUTPUT_PATH}"
  exit 0
fi

SBATCH_ARGS=(
  --job-name=oracle_init
  --gres="${ORACLE_GRES}"
  --cpus-per-task="${ORACLE_CPUS_PER_TASK}"
  --mem="${ORACLE_MEM}"
  --time="${ORACLE_TIME}"
  --output="${SCRIPT_DIR}/../logs/%x_%j.out"
  --error="${SCRIPT_DIR}/../logs/%x_%j.err"
)
[ -z "${ORACLE_PARTITION}" ] || SBATCH_ARGS+=(--partition="${ORACLE_PARTITION}")
[ -z "${ORACLE_QOS}" ] || SBATCH_ARGS+=(--qos="${ORACLE_QOS}")
[ -z "${ORACLE_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${ORACLE_NODELIST}")
[ -z "${ORACLE_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${ORACLE_EXCLUDE_NODES}")

mkdir -p "${SCRIPT_DIR}/../logs"
echo "Submit oracle init-state matching"
echo "  mode    : ${ORACLE_MODE}"
echo "  dataset : ${ORACLE_LEROBOT_DATASET}"
echo "  output  : ${ORACLE_OUTPUT_PATH}"
echo "  slurm   : partition=${ORACLE_PARTITION:-auto} qos=${ORACLE_QOS:-default} ${ORACLE_GRES}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  launch  : srun in allocation ${SLURM_JOB_ID}"
  ORACLE_MATCHING_DIR="${SCRIPT_DIR}" ORACLE_MATCHING_CONFIG="${CONFIG_PATH}" \
    srun "${SCRIPT_DIR}/src/oracle_matching.sbatch"
else
  echo "  launch  : sbatch"
  ORACLE_MATCHING_DIR="${SCRIPT_DIR}" ORACLE_MATCHING_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/oracle_matching.sbatch"
fi
