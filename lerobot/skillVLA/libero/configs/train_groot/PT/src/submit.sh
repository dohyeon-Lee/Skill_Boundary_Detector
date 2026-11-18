#!/usr/bin/env bash
# Shared PT/FT submission, using the same config snapshots and Slurm/local launcher as pi05.
set -euo pipefail
SUBMIT_DIR="$1"
CONFIG_PATH="$2"
shift 2
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SRC_DIR}/../../.." && pwd)"
REPO_DIR="$(cd "${CONFIGS_DIR}/../../../.." && pwd)"
BOOTSTRAP_PYTHON="${GROOT_BOOTSTRAP_PYTHON:-${REPO_DIR}/.venv_groot_lerobot/bin/python}"
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
  echo "Groot Python not found: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
fi
OPTIONS=()
if [ "${SMOKE}" = true ]; then OPTIONS+=(--smoke); fi
if [ "${DRY_RUN}" = true ]; then
  exec "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/groot_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}"
fi
source "${CONFIGS_DIR}/src/snapshot_config.sh"
source "${CONFIGS_DIR}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/groot_config.py" --config "${CONFIG_PATH}" "${OPTIONS[@]}" --shell --check --freeze)"
eval "${EXPORTS}"
export GROOT_TRAIN_CONFIG="${CONFIG_PATH}"
export GROOT_SRC_DIR="${SRC_DIR}"
export GROOT_SMOKE="${SMOKE}"
export GROOT_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}"
SBATCH_ARGS=(
  --job-name="groot_${GROOT_STAGE^^}"
  --gres="${GROOT_GRES}" --cpus-per-task="${GROOT_CPUS}"
  --mem="${GROOT_MEM}" --time="${GROOT_TIME}"
)
if [ -n "${GROOT_PARTITION}" ]; then SBATCH_ARGS+=(--partition="${GROOT_PARTITION}"); fi
if [ -n "${GROOT_QOS}" ]; then SBATCH_ARGS+=(--qos="${GROOT_QOS}"); fi
if [ -n "${GROOT_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${GROOT_NODELIST}"); fi
if [ -n "${GROOT_EXCLUDE}" ]; then SBATCH_ARGS+=(--exclude="${GROOT_EXCLUDE}"); fi
cd "${SUBMIT_DIR}"
mkdir -p logs
echo "Submit Groot ${GROOT_STAGE^^} (smoke=${SMOKE})"
echo "  dataset: ${GROOT_DATASET_DIR}"
echo "  output : ${GROOT_OUTPUT_DIR}"
submit_job "${SBATCH_ARGS[@]}" "${SUBMIT_DIR}/src/train.sbatch"
