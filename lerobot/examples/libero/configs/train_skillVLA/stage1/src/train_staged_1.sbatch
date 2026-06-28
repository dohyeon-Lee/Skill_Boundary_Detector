#!/usr/bin/env bash
# SkillVLA Stage-1 — STAGED phase 1-1 (single run) → {run}/1-1/.
# Expert-vision + action expert, NO connector, late(or plain)-weighted, hold boundary, init = pi05.
# Run this FIRST; inspect 1-1, then run train_staged_2.sbatch warm-starting from a chosen checkpoint.
#SBATCH --job-name=stage1_staged_1
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

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --mode staged --shell)"
source "${SRC_DIR}/_train_phase.sh"

echo "START STAGE-1 STAGED phase 1-1 → ${PHASE1_OUTPUT_DIR}  (steps=${STEPS}, weighting=${STAGED_PHASE1_WEIGHTING})"
train_phase "${PHASE1_OUTPUT_DIR}" "${STEPS}" "false" "${STAGED_PHASE1_WEIGHTING}" "false" "hold" "${PI_BASE}" "plain"
echo "DONE 1-1 → ${PHASE1_OUTPUT_DIR}   (next: train_staged_2.sbatch)"
