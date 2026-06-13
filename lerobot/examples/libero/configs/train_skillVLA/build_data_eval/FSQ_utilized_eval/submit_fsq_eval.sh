#!/usr/bin/env bash
# run_fsq_eval.sh 전체를 slurm 잡으로 (②의 DINO 추론 가속용 GPU 1개).
# Usage: ./submit_fsq_eval.sh [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/fsq_utilized_eval_config.py" --shell)"

cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=FSQ_utilized_eval
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres=gpu:1
  --cpus-per-task=8
  --mem=32G
  --time=4:00:00
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

sbatch "${SBATCH_ARGS[@]}" --wrap="${SCRIPT_DIR}/run_fsq_eval.sh ${1:-}"
