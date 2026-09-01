#!/usr/bin/env bash
# Submit the three sequential CALVIN split conversions as one Slurm GPU job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CALVIN_LONG_HORIZON_SPLIT_CONFIG:-${SCRIPT_DIR}/calvin_task_split_config.yaml}"
CONFIG_PY="${SCRIPT_DIR}/src/calvin_task_split_config.py"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"

# Freeze both the selection YAML and the shared CALVIN conversion YAML. The
# latter also carries the global dataset root and Slurm settings.
ORIGINAL_SETTINGS="$(PYTHONDONTWRITEBYTECODE=1 "${BOOTSTRAP_PYTHON}" \
  "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
eval "${ORIGINAL_SETTINGS}"
CONVERSION_CONFIG_PATH="$(snapshot_config "${CALVIN_SPLIT_CONVERSION_CONFIG}")"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

RESOLVED_SETTINGS="$(CALVIN_CONVERSION_CONFIG="${CONVERSION_CONFIG_PATH}" \
  PYTHONDONTWRITEBYTECODE=1 "${BOOTSTRAP_PYTHON}" \
  "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
eval "${RESOLVED_SETTINGS}"

if [ ! -d "${CALVIN_SPLIT_SOURCE_DIR}" ]; then
  echo "CALVIN source split not found: ${CALVIN_SPLIT_SOURCE_DIR}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
mkdir -p logs
SBATCH_ARGS=(
  --job-name=split_calvin
  --partition="${CALVIN_SPLIT_PARTITION}"
  --qos="${CALVIN_SPLIT_QOS}"
  --gres="${CALVIN_SPLIT_GRES}"
  --cpus-per-task="${CALVIN_SPLIT_CPUS_PER_TASK}"
  --mem="${CALVIN_SPLIT_MEM}"
  --time="${CALVIN_SPLIT_TIME}"
  --requeue
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
[ -n "${CALVIN_SPLIT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${CALVIN_SPLIT_NODELIST}")
[ -n "${CALVIN_SPLIT_EXCLUDE_NODES}" ] && \
  SBATCH_ARGS+=(--exclude="${CALVIN_SPLIT_EXCLUDE_NODES}")

WRAP="CALVIN_CONVERSION_CONFIG=$(printf %q "${CONVERSION_CONFIG_PATH}") CALVIN_LONG_HORIZON_SPLIT_CONFIG=$(printf %q "${CONFIG_PATH}") ${SCRIPT_DIR}/build_calvin_task_split.sh"
echo "== Submit CALVIN long-horizon three-way split =="
echo "  source : ${CALVIN_SPLIT_SOURCE_DIR}"
echo "  output : ${CALVIN_SPLIT_OUTPUT_ROOT}"
echo "  slurm  : partition=${CALVIN_SPLIT_PARTITION} qos=${CALVIN_SPLIT_QOS}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
