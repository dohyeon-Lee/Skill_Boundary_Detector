#!/usr/bin/env bash
# Submit unified SkillVLA Stage-1 Arch0--4 training (policy.type=skill_expert).
# Usage: ./submit_train.sh [arch0|arch0_1|arch0_2|arch0_2_sep|arch0_3|arch1_1|arch1_2|arch1_3|arch2_1|arch2_2|arch3|arch4]
#   (login) resolve config + check the skillvla dataset → sbatch train.sbatch
# The skillvla dataset comes from configs/train_skillVLA/build_data (run that first if missing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage1
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SCRIPT_DIR}/stage1_train_config.yaml}"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [arch0|arch0_1|arch0_2|arch0_2_sep|arch0_3|arch1_1|arch1_2|arch1_3|arch2_1|arch2_2|arch3|arch4]" >&2
  exit 2
fi
ARCHITECTURE_OVERRIDE="${1:-${STAGE1_ARCHITECTURE_OVERRIDE:-}}"
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  case "${ARCHITECTURE_OVERRIDE}" in
    arch0|arch0_1|arch0_2|arch0_2_sep|arch0_3|arch1_1|arch1_2|arch1_3|arch2_1|arch2_2|arch3|arch4) ;;
    *)
      echo "Unknown Stage-1 architecture: ${ARCHITECTURE_OVERRIDE}" >&2
      exit 2
      ;;
  esac
fi

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  CONFIG_ARGS+=(--architecture "${ARCHITECTURE_OVERRIDE}")
fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" "${CONFIG_ARGS[@]}"
)"; then
  echo "Stage-1 configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"
: "${SKILLVLA_DATASET_DIR:?Stage-1 bootstrap did not export SKILLVLA_DATASET_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
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
if [ -n "${TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit Stage-1 ${ARCHITECTURE_LABEL} (skill_expert)"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  expert   : full fine-tuning from ${PI_BASE}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

STAGE1_TRAIN_CONFIG="${CONFIG_PATH}" \
STAGE1_ARCHITECTURE_OVERRIDE="${ARCHITECTURE_OVERRIDE}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
