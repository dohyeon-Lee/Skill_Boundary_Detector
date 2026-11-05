#!/usr/bin/env bash
# IID and cyclic share the CURRENT train_pi05 PT command and recipe. No auto-cleanup.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT="$(cd "${CONFIGS}/../../../.." && pwd)"
MODE="${1:-iid}"
case "${MODE}" in iid|cyclic) ;; *) echo 'Usage: bash submit_train.sh [iid|cyclic]' >&2; exit 2 ;; esac
CONFIG_PATH="${TRAIN_CYCLIC_CONFIG:-${SCRIPT_DIR}/${MODE}_config.yaml}"
RESOLVER="${SCRIPT_DIR}/src/config.py"
PYTHON="${PROJECT}/.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

source "${CONFIGS}/src/submit_job.sh"
# Snapshot the inherited PT YAML as well as the local/global/server config, so later
# source-PT edits cannot alter a queued experiment. Dry-run has no filesystem writes.
if [ "${TRAIN_CYCLIC_DRY_RUN:-0}" != 1 ]; then
  CONFIG_PATH="$("${PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --snapshot)"
fi
EXPORTS="$("${PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --shell)"
eval "${EXPORTS}"
echo "pi05 ${CYCLIC_MODE}: ${PT_RUN_NAME}"
echo "  source: ${CYCLIC_SOURCE_CONFIG}"
echo "  batch/workers/CPU: ${PT_BATCH_SIZE}/${PT_NUM_WORKERS}/${PT_CPUS_PER_TASK}"
echo "  XYZ grounding: ${PT_PROPRIO_GROUNDING}; steps=${PT_STEPS}; LR=${PT_LR_MODE}"
echo "  gradient checkpointing: ${PT_GRADIENT_CHECKPOINTING}"
echo "  W&B project: ${PT_WANDB_PROJECT}; probes: ${CYCLIC_PROBE_BATCHES_PER_GROUP} batches/group every ${CYCLIC_PHASE_STEPS} steps; gradient group=${CYCLIC_PROBE_GRAD_GROUP}"
echo "  qos=${PT_QOS}; partitions=${PT_PARTITION}"
echo "  output: ${PT_OUTPUT_DIR}"
if [ "${TRAIN_CYCLIC_DRY_RUN:-0}" = 1 ]; then
  echo 'Dry run: no job submitted, no snapshot/output created.'
  exit 0
fi
mkdir -p "${SCRIPT_DIR}/logs"
ARGS=(--job-name="pi05_${CYCLIC_MODE}" --partition="${PT_PARTITION}" --qos="${PT_QOS}"
  --gres="${PT_GRES}" --cpus-per-task="${PT_CPUS_PER_TASK}" --mem="${PT_MEM}" --time="${PT_TIME}"
  --output="${SCRIPT_DIR}/logs/%x_%j.out" --error="${SCRIPT_DIR}/logs/%x_%j.err")
[ -z "${PT_NODELIST}" ] || ARGS+=(--nodelist="${PT_NODELIST}")
[ -z "${PT_EXCLUDE_NODES}" ] || ARGS+=(--exclude="${PT_EXCLUDE_NODES}")
# Reuse the original batch script INCLUDING every accelerate/CLI flag. The only
# opt-in there is a different resolver; both modes enter the native trainer with diagnostics.
cd "${CONFIGS}/train_pi05/PT"
TRAIN_PI05_CONFIG="${CONFIG_PATH}" TRAIN_PI05_CONFIG_RESOLVER="${RESOLVER}" PI05_PRESERVE_INCOMPLETE_OUTPUT=1 \
  submit_job "${ARGS[@]}" "${CONFIGS}/train_pi05/PT/src/train.sbatch"
