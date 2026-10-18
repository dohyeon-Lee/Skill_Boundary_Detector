#!/usr/bin/env bash
# Shared PT/FT submission, using the same config snapshots and Slurm/local launcher as pi05.
set -euo pipefail
SUBMIT_DIR="$1"
CONFIG_PATH="$2"
shift 2
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SRC_DIR}/../../.." && pwd)"
REPO_DIR="$(cd "${CONFIGS_DIR}/../../../.." && pwd)"
BOOTSTRAP_PYTHON="${MOLMOACT2_BOOTSTRAP_PYTHON:-${REPO_DIR}/.venv_molmoact2_lerobot/bin/python}"
SMOKE=false
DRY_RUN=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --config) CONFIG_PATH="${2:?--config needs a path}"; shift 2 ;;
    -h|--help)
      echo "Usage: ./submit_train.sh [--smoke] [--dry-run] [--config path]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  echo "MolmoAct2 Python not found: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
fi
OPTIONS=()
if [ "${SMOKE}" = true ]; then OPTIONS+=(--smoke); fi
if [ "${DRY_RUN}" = true ]; then
  exec "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/molmoact2_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}"
fi
source "${CONFIGS_DIR}/src/snapshot_config.sh"
source "${CONFIGS_DIR}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/molmoact2_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}" --shell --check --freeze)"
eval "${EXPORTS}"
export MOLMOACT2_TRAIN_CONFIG="${CONFIG_PATH}"
export MOLMOACT2_SRC_DIR="${SRC_DIR}"
export MOLMOACT2_SMOKE="${SMOKE}"
export MOLMOACT2_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}"
SBATCH_ARGS=(
  --job-name="molmoact2_${MOLMOACT2_STAGE^^}"
  --gres="${MOLMOACT2_GRES}" --cpus-per-task="${MOLMOACT2_CPUS}"
  --mem="${MOLMOACT2_MEM}" --time="${MOLMOACT2_TIME}"
)
if [ -n "${MOLMOACT2_PARTITION}" ]; then SBATCH_ARGS+=(--partition="${MOLMOACT2_PARTITION}"); fi
if [ -n "${MOLMOACT2_QOS}" ]; then SBATCH_ARGS+=(--qos="${MOLMOACT2_QOS}"); fi
if [ -n "${MOLMOACT2_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${MOLMOACT2_NODELIST}"); fi
if [ -n "${MOLMOACT2_EXCLUDE}" ]; then SBATCH_ARGS+=(--exclude="${MOLMOACT2_EXCLUDE}"); fi
cd "${SUBMIT_DIR}"
mkdir -p logs
echo "Submit MolmoAct2 ${MOLMOACT2_STAGE^^} (smoke=${SMOKE})"
echo "  dataset: ${MOLMOACT2_DATASET_DIR}"
echo "  output : ${MOLMOACT2_OUTPUT_DIR}"
submit_job "${SBATCH_ARGS[@]}" "${SUBMIT_DIR}/src/train.sbatch"
