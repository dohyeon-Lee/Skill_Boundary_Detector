#!/usr/bin/env bash
# Queue run_tf_compare.sh as a slurm job (GPU 1개; teacher_forced가 모델을 GPU로 로드).
# Usage: ./submit_tf_compare.sh [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/tf_compare_config.py" --shell)"

cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=S1_TF_compare
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

sbatch "${SBATCH_ARGS[@]}" --wrap="${SCRIPT_DIR}/run_tf_compare.sh ${1:-}"
