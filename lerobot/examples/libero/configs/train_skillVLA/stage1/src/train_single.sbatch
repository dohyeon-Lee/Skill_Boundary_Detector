#!/usr/bin/env bash
# SkillVLA Stage-1 — SINGLE-STAGE (CFG-style) → {base}/single/<oracle>.
# One run: Oracle ON from scratch, NO freeze. CFG A/B dropout (oracle_dropout_p) makes the VSA base
# stand alone and forces r to be the residual — same equilibrium as staged, fewer stages. Init = pi05.
#SBATCH --job-name=stage1_single
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

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --shell)"
source "${SRC_DIR}/_train_phase.sh"

echo "START STAGE-1 SINGLE → ${SINGLE_DIR}  (steps=${STEPS})"
#           OUT              NAME               STEPS      USE_ORACLE FREEZE_VSA PRETRAINED
train_phase "${SINGLE_DIR}" "${SINGLE_NAME}"  "${STEPS}" "true"     "false"    "${PI_BASE}"
echo "DONE SINGLE → ${SINGLE_DIR}"
