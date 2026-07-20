#!/usr/bin/env bash
# Submit cycle-PT eval using Slurm settings from ./cycle_eval_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_PI05_CYCLE_EVAL_CONFIG:-${SCRIPT_DIR}/cycle_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_cycle_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -n "${EVAL_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -n "${EVAL_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs outputs

echo "Submit cycle-PT eval"
echo "  policy  : ${EVAL_POLICY_PATH}"
echo "  target  : ${EVAL_TARGET_TASK} (ids ${EVAL_TASK_IDS})"
echo "  slurm   : partition=${EVAL_PARTITION} nodelist=${EVAL_NODELIST:-<none>} exclude=${EVAL_EXCLUDE_NODES:-<none>}"

TRAIN_PI05_CYCLE_EVAL_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/eval.sbatch"
