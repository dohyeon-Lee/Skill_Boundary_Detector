#!/usr/bin/env bash
# Submit the Stage-1 INPUT-influence probe (state / skill / progress, vs pi05) — offline, no simulator.
#   (login) resolve config + check checkpoints → sbatch probe.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # input_probe
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${INPUT_PROBE_CONFIG:-${SCRIPT_DIR}/input_probe_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/input_probe_config.py" --config "${CONFIG_PATH}" --shell)"

if [ ! -e "${DATASET_DIR}" ]; then
  echo "skillvla dataset not found: ${DATASET_DIR}  (check the first cond target's model_dir)" >&2
  exit 1
fi
# Per-target checkpoint existence is validated inside input_probe.py (clear error per missing path).

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
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

echo "Submit INPUT PROBE"
echo "  targets: ${TARGETS_JSON}"
echo "  dataset: ${DATASET_DIR}"
echo "  out    : ${OUTPUT_DIR}"
echo "  slurm  : partition=${EVAL_PARTITION} qos=${EVAL_QOS} mem=${EVAL_MEM}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun (reusing allocation ${SLURM_JOB_ID})"
  INPUT_PROBE_DIR="${SCRIPT_DIR}" INPUT_PROBE_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/probe.sbatch"
else
  echo "  mode   : sbatch (new job)"
  INPUT_PROBE_DIR="${SCRIPT_DIR}" INPUT_PROBE_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/probe.sbatch"
fi
