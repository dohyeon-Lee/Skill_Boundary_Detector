#!/usr/bin/env bash
# Submit SkillVLA Stage-3 training over a frozen Stage-0 or Stage-2 model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage3
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE3_TRAIN_CONFIG:-${SCRIPT_DIR}/stage3_train_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Freeze the resolved env to a per-submit snapshot the JOB sources verbatim (no job-side emitter re-run
# on a possibly deleted/edited yaml). Failure surfaces HERE at submit, not as a job-side traceback.
mkdir -p "${SCRIPT_DIR}/logs"
STAGE3_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/stage3_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage3_train_config.py" --config "${CONFIG_PATH}" --shell > "${STAGE3_ENV_SNAPSHOT}"
source "${STAGE3_ENV_SNAPSHOT}"
export STAGE3_ENV_SNAPSHOT

if [ ! -e "${PARENT_CHECKPOINT_PATH}" ]; then
  echo "${PARENT_STAGE} checkpoint not found: ${PARENT_CHECKPOINT_PATH}" >&2
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
[ -n "${TRAIN_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
[ -n "${TRAIN_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit SkillVLA STAGE-3"
echo "  parent    : ${PARENT_STAGE} / ${PARENT_RUN_NAME} / ${PARENT_CHECKPOINT}"
echo "  warm-start: ${PARENT_CHECKPOINT_PATH}"
echo "  FSQ       : ${FSQ_CKPT} (${FSQ_SOURCE})"
echo "  dataset   : ${SKILLVLA_DATASET_DIR}"
echo "  output    : ${PT_OUTPUT_DIR}"
echo "  slurm     : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} nodelist=${TRAIN_NODELIST:-<none>} exclude=${TRAIN_EXCLUDE_NODES:-<none>}"

STAGE3_TRAIN_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
