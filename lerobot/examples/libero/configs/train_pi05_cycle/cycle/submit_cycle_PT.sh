#!/usr/bin/env bash
# Submit block-cyclic pi05 PT using Slurm settings from ./cycle_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_PI05_CYCLE_CONFIG:-${SCRIPT_DIR}/cycle_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_cycle_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --partition="${PT_PARTITION}"
  --qos="${PT_QOS}"
  --gres="${PT_GRES}"
  --cpus-per-task="${PT_CPUS_PER_TASK}"
  --mem="${PT_MEM}"
  --time="${PT_TIME}"
)
[ -n "${PT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${PT_NODELIST}")
[ -n "${PT_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${PT_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit pi05 cyclic PT"
echo "  dataset : ${PT_DATASET_DIR}"
echo "  output  : ${PT_OUTPUT_DIR}"
echo "  cycle   : groups=${CYCLE_N_GROUPS} phase=${CYCLE_PHASE_STEPS} lambda=${CYCLE_DELTA_LAMBDA} beta=${CYCLE_REPTILE_BETA}"
echo "  slurm   : partition=${PT_PARTITION} nodelist=${PT_NODELIST:-<none>} exclude=${PT_EXCLUDE_NODES:-<none>}"

TRAIN_PI05_CYCLE_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/cycle_PT.sbatch"
