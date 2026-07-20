#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE0_PRETRAIN_CONFIG:-${SCRIPT_DIR}/stage0_pretrain_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

mkdir -p "${SCRIPT_DIR}/logs"
STAGE0_PRETRAIN_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/stage0_pretrain_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage0_pretrain_config.py" \
  --config "${CONFIG_PATH}" --shell > "${STAGE0_PRETRAIN_ENV_SNAPSHOT}"
source "${STAGE0_PRETRAIN_ENV_SNAPSHOT}"

for required in "${SKILLVLA_DATASET_DIR}" "${PI_BASE}" "${TOKENIZER_PATH}" "${FSQ_CKPT}" \
                "${PRETRAIN_CHECKPOINT_PATH}" "${TRANSITION_PACK}" "${PRETRAIN_TARGET_PACK}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing Stage-0 prerequisite: ${required}" >&2
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
echo "Submit SkillVLA Stage-0-pretrain"
echo "  run     : ${PT_RUN_NAME}"
echo "  dataset : ${SKILLVLA_DATASET_DIR}"
echo "  FSQ     : ${FSQ_CKPT}"
echo "  VLM     : ${PRETRAIN_CHECKPOINT_PATH} (${PRETRAIN_TRAINING_MODE})"
echo "  AR      : skill=on FAST=${AR_FAST_LOSS} batch=${AR_BATCH_SIZE}"
echo "  output  : ${PT_OUTPUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode    : srun (allocation ${SLURM_JOB_ID})"
  STAGE0_PRETRAIN_CONFIG="${CONFIG_PATH}" STAGE0_PRETRAIN_ENV_SNAPSHOT="${STAGE0_PRETRAIN_ENV_SNAPSHOT}" \
    srun "${SRC_DIR}/train.sbatch"
else
  echo "  mode    : sbatch"
  STAGE0_PRETRAIN_CONFIG="${CONFIG_PATH}" STAGE0_PRETRAIN_ENV_SNAPSHOT="${STAGE0_PRETRAIN_ENV_SNAPSHOT}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
fi
