#!/usr/bin/env bash
#SBATCH --job-name=MimicGen_pi05_FT
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:-$(pwd)}"
CODE_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export TRAIN_PI05_CONFIG_RESOLVER="${CODE_ROOT}/lerobot/skillVLA/mimicgen/configs/train_pi05/FT/src/config.py"
export PI05_FT_TRAIN_BIN="${CODE_ROOT}/lerobot/skillVLA/mimicgen/configs/train_pi05/FT/src/train.py"
export PYTHONPATH="${CODE_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${CODE_ROOT}/lerobot/skillVLA/libero/configs/train_pi05/FT/src/train.sbatch" "$@"
