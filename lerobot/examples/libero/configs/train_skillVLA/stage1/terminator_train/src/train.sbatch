#!/usr/bin/env bash
# Standalone FSQ-terminator training (disjoint from any policy). Resolves submit plumbing via
# terminator_train_config.py --shell, then runs train_terminator.py (which reads the yaml directly).
#SBATCH --job-name=term
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR (the terminator_train dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TERMINATOR_TRAIN_CONFIG:-${SUBMIT_DIR}/terminator_train_config.yaml}"

# terminator_train is 7 levels below project_root (…/stage1/terminator_train).
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/terminator_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export PYTHONUNBUFFERED=1       # live stdout in the Slurm log (else prints buffer + look "stuck")
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START TERMINATOR TRAIN"
echo "Node : ${SLURMD_NODENAME:-local}"
echo "FSQ  : ${FSQ_PATH}"
echo "data : ${SKILLVLA_DATASET_DIR}   dino: ${DINO_PATH}"
nvidia-smi || true

for f in "${FSQ_PATH}" "${DINO_PATH}" "${SKILLVLA_DATASET_DIR}"; do
  if [ ! -e "${f}" ]; then echo "Missing: ${f}" >&2; exit 1; fi
done

python "${SRC_DIR}/train_terminator.py" --config "${CONFIG_PATH}"
echo "DONE"
