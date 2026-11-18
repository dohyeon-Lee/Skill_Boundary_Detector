#!/usr/bin/env bash
#SBATCH --job-name=molmoact2_PT
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
SRC_DIR="${MOLMOACT2_SRC_DIR:-${SUBMIT_DIR}/src}"
exec bash "${SRC_DIR}/run_train.sh" \
  "${MOLMOACT2_TRAIN_CONFIG:-${SUBMIT_DIR}/molmoact2_PT_config.yaml}" "${MOLMOACT2_SMOKE:-false}"
