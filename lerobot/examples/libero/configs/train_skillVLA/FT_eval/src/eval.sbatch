#!/usr/bin/env bash
# Direct-entry wrapper for the maintained Stage-2 closed-loop evaluator.
# submit_eval.sh normally submits that evaluator directly.
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

export STAGE2_EVAL_DIR="${FT_EVAL_DIR}/../stage2_eval"
export STAGE2_EVAL_CONFIG="${FT_EVAL_CONFIG:-${FT_EVAL_DIR}/ft_eval_config.yaml}"
export STAGE2_EVAL_CONFIG_RESOLVER="${FT_EVAL_DIR}/src/ft_eval_config.py"
export STAGE2_EVAL_LOG_DIR="${STAGE2_EVAL_LOG_DIR:-${FT_EVAL_DIR}/logs}"
export STAGE2_EVAL_VENV_LABEL="${STAGE2_EVAL_VENV_LABEL:-FT eval venv}"

exec "${STAGE2_EVAL_DIR}/src/eval.sbatch"
