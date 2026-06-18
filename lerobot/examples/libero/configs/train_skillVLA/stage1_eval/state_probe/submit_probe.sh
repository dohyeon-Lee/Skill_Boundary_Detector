#!/usr/bin/env bash
# Submit the Stage-1 STATE-influence probe (pi05 vs SkillVLA-cond) — offline, no simulator.
#   (login) resolve config + check checkpoints → sbatch probe.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # state_probe
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STATE_PROBE_CONFIG:-${SCRIPT_DIR}/state_probe_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/state_probe_config.py" --config "${CONFIG_PATH}" --shell)"

if [ ! -d "${COND_POLICY_PATH}" ]; then
  echo "cond (stage-1) checkpoint not found: ${COND_POLICY_PATH}" >&2
  echo "Train Stage-1 first (configs/train_skillVLA/stage1) or fix cond_model_dir/cond_checkpoint." >&2
  exit 1
fi
if [ -n "${PI05_POLICY_PATH}" ] && [ ! -d "${PI05_POLICY_PATH}" ]; then
  echo "pi05 checkpoint not found: ${PI05_POLICY_PATH}  (fix pi05_model_dir/pi05_checkpoint or blank it)" >&2
  exit 1
fi

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

echo "Submit STATE PROBE (pi05 vs cond)"
echo "  cond   : ${COND_POLICY_PATH}"
echo "  pi05   : ${PI05_POLICY_PATH:-(none)}"
echo "  dataset: ${DATASET_DIR}"
echo "  out    : ${OUTPUT_DIR}"
echo "  slurm  : partition=${EVAL_PARTITION} qos=${EVAL_QOS} mem=${EVAL_MEM}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun (reusing allocation ${SLURM_JOB_ID})"
  STATE_PROBE_DIR="${SCRIPT_DIR}" STATE_PROBE_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/probe.sbatch"
else
  echo "  mode   : sbatch (new job)"
  STATE_PROBE_DIR="${SCRIPT_DIR}" STATE_PROBE_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/probe.sbatch"
fi
