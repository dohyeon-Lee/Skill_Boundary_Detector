#!/usr/bin/env bash
# Submit the SkillVLA Stage-1 OFFLINE teacher-forced eval (no simulator).
# Uses the SAME stage1_eval_config.yaml as the oracle eval: model_dir + checkpoint pick the
# policy, and the skillvla dataset comes from the run_dir the model trained on.
# Optional env knobs: TF_N_FRAMES (512), TF_BATCH_SIZE (32), TF_N_SWAP (4).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage1_eval
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" --config "${CONFIG_PATH}" --shell)"

for p in "${POLICY_PATH}" "${SKILL_LABEL_DATASET_DIR}"; do
  if [ ! -e "${p}" ]; then
    echo "Missing artifact: ${p}" >&2
    exit 1
  fi
done

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

echo "Submit Stage-1 teacher-forced eval (offline)"
echo "  policy  : ${POLICY_PATH}"
echo "  dataset : ${SKILL_LABEL_DATASET_DIR}"
echo "  out     : ${EVAL_OUT_DIR}/teacher_forced"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job step.
  echo "  mode    : srun (reusing allocation ${SLURM_JOB_ID})"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/teacher_forced.sbatch"
else
  echo "  mode    : sbatch (new job)"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/teacher_forced.sbatch"
fi
