#!/usr/bin/env bash
# SkillVLA Stage-1 — JOINT + PLAIN (single run). All modules together, uniform/action_weighting loss.
# Experiment is fixed by THIS file (the yaml is not edited per-experiment).
#SBATCH --job-name=stage1_joint_plain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SUBMIT_DIR}/stage1_train_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"; [ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --mode joint --loss-mode plain --shell)"
source "${SRC_DIR}/_train_phase.sh"

echo "START STAGE-1 JOINT+PLAIN → ${PT_OUTPUT_DIR}"
train_phase "${PT_OUTPUT_DIR}" "${STEPS}" "${USE_CONNECTOR}" "plain" "false" "${BOUNDARY_MODE}" "${PI_BASE}" "${LOSS_MODE}"
echo "DONE → ${PT_OUTPUT_DIR}"
