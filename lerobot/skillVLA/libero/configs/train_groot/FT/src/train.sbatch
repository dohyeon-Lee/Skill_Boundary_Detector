#!/usr/bin/env bash
#SBATCH --job-name=groot_FT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SRC_DIR="${GROOT_SRC_DIR:-${SUBMIT_DIR}/../PT/src}"
exec bash "${SRC_DIR}/run_train.sh" \
  "${GROOT_TRAIN_CONFIG:-${SUBMIT_DIR}/groot_FT_config.yaml}" "${GROOT_SMOKE:-false}"
