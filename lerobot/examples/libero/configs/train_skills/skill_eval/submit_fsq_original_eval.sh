#!/usr/bin/env bash
# Submit FSQ-original checkpoint evaluations: one job per (run, checkpoint).
# Runs/checkpoints come from fsq_original_eval_config.yaml; each run's skills
# dir is resolved from its own fsq_original_meta.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FSQ_ORIG_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_original_eval_config.yaml}"

# Freeze the config so queued jobs ignore later edits (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_original_eval_config.py" --config "${CONFIG_PATH}" --shell)"
eval "${RESOLVED_SETTINGS}"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

IFS=',' read -r -a RUNS <<< "${FSQ_ORIG_EVAL_RUNS}"
read -r -a CHECKPOINTS <<< "${FSQ_ORIG_EVAL_CHECKPOINTS}"

for RUN in "${RUNS[@]}"; do
  RUN_DIR="${FSQ_OUTPUTS_ROOT}/${RUN}"
  if [ ! -f "${RUN_DIR}/fsq_original_meta.json" ]; then
    echo "Skip ${RUN}: not an FSQ-original run (fsq_original_meta.json missing)" >&2
    continue
  fi
  # Fail fast on an unresolvable skills dir before queueing anything.
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_original_eval_config.py" --config "${CONFIG_PATH}" --resolve-skills "${RUN_DIR}" > /dev/null
  for TAG in "${CHECKPOINTS[@]}"; do
    if [ ! -f "${RUN_DIR}/FSQ_${TAG}.pt" ]; then
      echo "Skip ${RUN} ${TAG}: checkpoint missing" >&2
      continue
    fi
    echo "Submit FSQ-original eval: ${RUN} ${TAG}"
    FSQ_ORIG_EVAL_CONFIG="${CONFIG_PATH}" FSQ_RUN_NAME="${RUN}" FSQ_EPOCH_TAG="${TAG}" \
      sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/fsq_original_eval.sbatch"
  done
done
