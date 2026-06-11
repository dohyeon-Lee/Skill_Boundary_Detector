#!/usr/bin/env bash
# Submit the teacher-forced CHECKPOINT SWEEP for the Stage-1 run in stage1_eval_config.yaml
# (model_dir picks the run; the yaml's checkpoint field is ignored — all numeric checkpoints
# run by default). Skips checkpoints whose result already exists → resubmit as training
# progresses to extend the trend.
# Env knobs: TF_CKPTS="005000 025000 last", TF_N_FRAMES (512), TF_BATCH_SIZE (32), TF_N_SWAP (4).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage1_eval
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml.
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" --config "${CONFIG_PATH}" --shell)"

MODEL_ROOT="$(dirname "$(dirname "$(dirname "${POLICY_PATH}")")")"
if [ ! -d "${MODEL_ROOT}/checkpoints" ]; then
  echo "No checkpoints dir: ${MODEL_ROOT}/checkpoints" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="4:00:00"
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit Stage-1 teacher-forced SWEEP"
echo "  model  : $(basename "${MODEL_ROOT}")"
echo "  ckpts  : ${TF_CKPTS:-all numeric under ${MODEL_ROOT}/checkpoints}"
echo "  dataset: ${SKILL_LABEL_DATASET_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun (reusing allocation ${SLURM_JOB_ID})"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" TF_CKPTS="${TF_CKPTS:-}" \
    srun "${SRC_DIR}/teacher_forced_sweep.sbatch"
else
  echo "  mode   : sbatch (new job)"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" TF_CKPTS="${TF_CKPTS:-}" \
    sbatch "${SBATCH_ARGS[@]}" --export=ALL "${SRC_DIR}/teacher_forced_sweep.sbatch"
fi
