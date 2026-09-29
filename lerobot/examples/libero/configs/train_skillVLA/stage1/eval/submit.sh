#!/usr/bin/env bash
# Validate/snapshot the config and submit one GPU visualization job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_ATTENTION_EVAL_CONFIG:-${SCRIPT_DIR}/attention_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

_lib="${SCRIPT_DIR}"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

if ! EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/eval_config.py" --config "${CONFIG_PATH}" --shell)"; then
  echo "Stage-1 attention-eval config validation failed; no job was submitted." >&2
  exit 1
fi
eval "${EXPORTS}"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
if [ -n "${EVAL_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}"); fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}"); fi

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Stage-1 alignment maps: ${EVAL_OUTPUT_DIR}"
STAGE1_ATTENTION_EVAL_SRC_DIR="${SRC_DIR}" \
ATTENTION_EVAL_CONFIG="${CONFIG_PATH}" \
  submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/run.sbatch"
