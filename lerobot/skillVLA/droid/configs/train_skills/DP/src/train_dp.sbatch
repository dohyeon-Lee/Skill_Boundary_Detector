#!/usr/bin/env bash
#SBATCH --job-name=DROID_DP
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
exec bash "${PROJECT_ROOT:?}/lerobot/skillVLA/droid/configs/train_skills/src/run_training.sh" DP "$1"
