#!/usr/bin/env bash
# Offline front-vs-back SKILL loss eval (no sim) as a Slurm job. Runs run.sh (venv + PYTHONPATH + loss_eval.py)
# on one GPU. Args are forwarded via LOSS_EVAL_ARGS (set by submit.sh). Slurm resources come from submit.sh's
# --partition/--gres/... which override the #SBATCH defaults below.
#SBATCH --job-name=S1lossE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
HERE="${LOSS_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${HERE}/logs"
echo "START loss-eval  node=${SLURMD_NODENAME:-local}  args=${LOSS_EVAL_ARGS:-}"
nvidia-smi || true
# shellcheck disable=SC2086
exec "${HERE}/run.sh" ${LOSS_EVAL_ARGS:-}
