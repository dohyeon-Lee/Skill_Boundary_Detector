#!/usr/bin/env bash
# Run an FT evaluation with the shared Stage-2 closed-loop evaluator.
#SBATCH --job-name=FTeval
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

FT_EVAL_DIR="${FT_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${FT_EVAL_CONFIG:-${FT_EVAL_DIR}/ft_eval_config.yaml}"
BOOTSTRAP_PYTHON=/usr/bin/python3

# submit_eval.sh normally exports one frozen resolution. This branch keeps the
# sbatch file directly runnable as well.
if [ -z "${POLICY_PATH:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${FT_EVAL_DIR}/src/ft_eval_config.py" \
    --config "${CONFIG_PATH}" --shell)"
fi

# The shared script owns environment setup, CUDA guarding, evaluation, and
# chunk merging. All paths and panel specs above remain FT-resolved exports.
export STAGE2_EVAL_DIR="${FT_EVAL_DIR}/../stage2_eval"
export STAGE2_EVAL_CONFIG="${CONFIG_PATH}"
exec "${STAGE2_EVAL_DIR}/src/eval.sbatch"
