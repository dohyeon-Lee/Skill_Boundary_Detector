#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${PRETRAIN_CONFIG:-${SCRIPT_DIR}/pretrain_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
mkdir -p "${SCRIPT_DIR}/logs"
PRETRAIN_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/pretrain_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/pretrain_config.py" \
  --config "${CONFIG_PATH}" --shell > "${PRETRAIN_ENV_SNAPSHOT}"
source "${PRETRAIN_ENV_SNAPSHOT}"

for required in "${SKILLVLA_DATASET_DIR}" "${TRANSITION_PACK}" "${PRETRAIN_TARGETS}" \
                "${PI_BASE}" "${TEXT_TOKENIZER}" "${FAST_TOKENIZER}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing pretraining prerequisite: ${required}" >&2
    if [ "${required}" = "${FAST_TOKENIZER}" ] || [ "${required}" = "${PRETRAIN_TARGETS}" ]; then
      echo "Run ${SCRIPT_DIR}/prepare_fast_tokenizer.sh first." >&2
    fi
    exit 1
  fi
done

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
echo "Submit SkillVLA pretraining"
echo "  run     : ${PRETRAIN_RUN_NAME}"
echo "  mode    : ${TRAINING_MODE}"
echo "  FAST    : ${FAST_TOKENIZER}"
echo "  output  : ${PRETRAIN_OUTPUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  PRETRAIN_CONFIG="${CONFIG_PATH}" PRETRAIN_ENV_SNAPSHOT="${PRETRAIN_ENV_SNAPSHOT}" \
    srun "${SRC_DIR}/train.sbatch"
else
  PRETRAIN_CONFIG="${CONFIG_PATH}" PRETRAIN_ENV_SNAPSHOT="${PRETRAIN_ENV_SNAPSHOT}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
fi
