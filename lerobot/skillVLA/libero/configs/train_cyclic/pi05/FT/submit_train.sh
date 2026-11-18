#!/usr/bin/env bash
# Dedicated replay-free FT of either an IID or a cyclic PT checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT="$(cd "${CONFIGS}/../../../.." && pwd)"
CONFIG_PATH="${TRAIN_CYCLIC_FT_CONFIG:-${SCRIPT_DIR}/ft_config.yaml}"
RESOLVER="${SCRIPT_DIR}/../src/ft_config.py"
PYTHON="${PROJECT}/.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3
source "${CONFIGS}/src/submit_job.sh"
if [ "${TRAIN_CYCLIC_FT_DRY_RUN:-0}" != 1 ]; then
  CONFIG_PATH="$("${PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --snapshot)"
fi
EXPORTS="$("${PYTHON}" "${RESOLVER}" --config "${CONFIG_PATH}" --shell)"
eval "${EXPORTS}"
echo "pi05 cyclic/IID FT: ${FT_RUN_NAME}"
echo "  source: ${FT_PRETRAINED_MODEL_PATH} (ready=${CYCLIC_FT_SOURCE_READY})"
echo "  dataset: ${FT_DATASET_DIR}"
if [ "${CYCLIC_FT_SOURCE_READY}" = true ]; then
  echo "  grounding: ${FT_PROPRIO_GROUNDING}; normalization: ${FT_NORMALIZATION_STATS}"
else
  echo "  grounding/freeze/horizon: pending source checkpoint; normalization: ${FT_NORMALIZATION_STATS}"
fi
echo "  freeze: expert=${FT_FREEZE_EXPERT} vision=${FT_FREEZE_VISION_ENCODER} language=${FT_FREEZE_LANGUAGE_MODEL}"
echo "  steps=${FT_STEPS}; lr=${FT_LR}; schedule=${FT_LR_MODE}; probe every=${CYCLIC_FT_PROBE_EVERY}"
echo "  output: ${FT_OUTPUT_DIR}"
if [ "${TRAIN_CYCLIC_FT_DRY_RUN:-0}" = 1 ]; then
  echo 'Dry run: no job submitted, no snapshot/output created. Missing PT checkpoints are allowed here only.'
  exit 0
fi
mkdir -p "${SCRIPT_DIR}/logs"
ARGS=(--job-name=pi05_cyclic_FT --partition="${FT_PARTITION}" --qos="${FT_QOS}"
  --gres="${FT_GRES}" --cpus-per-task="${FT_CPUS_PER_TASK}" --mem="${FT_MEM}" --time="${FT_TIME}"
  --output="${SCRIPT_DIR}/logs/%x_%j.out" --error="${SCRIPT_DIR}/logs/%x_%j.err")
[ -z "${FT_NODELIST}" ] || ARGS+=(--nodelist="${FT_NODELIST}")
[ -z "${FT_EXCLUDE_NODES}" ] || ARGS+=(--exclude="${FT_EXCLUDE_NODES}")
cd "${CONFIGS}/train_pi05/FT"
TRAIN_PI05_CONFIG="${CONFIG_PATH}" TRAIN_PI05_CONFIG_RESOLVER="${RESOLVER}" \
  PI05_FT_TRAIN_BIN="${SCRIPT_DIR}/../src/train_ft.py" \
  submit_job "${ARGS[@]}" "${CONFIGS}/train_pi05/FT/src/train.sbatch"
