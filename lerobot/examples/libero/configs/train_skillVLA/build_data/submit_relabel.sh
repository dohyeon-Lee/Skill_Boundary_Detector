#!/usr/bin/env bash
# Submit only the predictor relabel post-processing stage. The source SkillVLA
# dataset must already have been completed by submit_build_all.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

mkdir -p "${SCRIPT_DIR}/logs"
SNAPSHOT_ENV="${SCRIPT_DIR}/logs/skillvla_relabel_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
CONFIG_ARGS=(--config "${CONFIG_PATH}" --relabel --shell)
if [ -n "${SOURCE_DATASET}" ]; then
  CONFIG_ARGS+=(--dataset "${SOURCE_DATASET}")
fi
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" "${CONFIG_ARGS[@]}" > "${SNAPSHOT_ENV}"
source "${SNAPSHOT_ENV}"

SBATCH_ARGS=(
  --partition="${SKILLVLA_PARTITION}"
  --qos="${SKILLVLA_QOS}"
  --gres="${SKILLVLA_GRES}"
  --cpus-per-task="${SKILLVLA_CPUS_PER_TASK}"
  --mem="${SKILLVLA_MEM}"
  --time="${SKILLVLA_TIME}"
)
[ -n "${SKILLVLA_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${SKILLVLA_NODELIST}")
[ -n "${SKILLVLA_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${SKILLVLA_EXCLUDE_NODES}")

echo "Submit SkillVLA predictor relabel"
echo "  source    : ${RELABEL_SOURCE_RUN_DIR}"
echo "  predictor : ${RELABEL_PREDICTOR_MODEL}/${RELABEL_PREDICTOR_CHECKPOINT}"
echo "  output    : ${RELABEL_OUTPUT_RUN_DIR}"

cd "${SCRIPT_DIR}"
env \
  TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" \
  SOURCE_DATA="${SOURCE_DATASET}" \
  SKILLVLA_ENV_SNAPSHOT="${SNAPSHOT_ENV}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/relabel_skillvla.sbatch"
