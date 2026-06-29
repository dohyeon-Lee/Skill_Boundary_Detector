#!/usr/bin/env bash
# SkillVLA Stage-1 — STAGED phase 1-2 → {base}/staged/1-2_<warmstart>_<oracle>.
# FREEZE the VSA base (vision+skill_proj+null), add the Oracle + r (CFG A/B dropout), warm-started
# from a CHOSEN 1-1 checkpoint (yaml staged_phase2_warmstart: last | step number | path).
# Run AFTER train_staged_1.sbatch.
#SBATCH --job-name=stage1_staged_2
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

if [ ! -e "${STAGED_2_WARMSTART}" ]; then
  echo "1-2 warm-start checkpoint not found: ${STAGED_2_WARMSTART}" >&2
  echo "  → run train_staged_1.sbatch first, or set staged_phase2_warmstart in the yaml." >&2
  exit 1
fi

echo "START STAGE-1 STAGED phase 1-2 → ${STAGED_1_2_DIR}  (steps=${STEPS}, warmstart=${STAGED_2_WARMSTART})"
#           OUT                 NAME                  STEPS      USE_ORACLE FREEZE_VSA PRETRAINED
train_phase "${STAGED_1_2_DIR}" "${STAGED_1_2_NAME}" "${STEPS}" "true"     "true"     "${STAGED_2_WARMSTART}"
echo "DONE 1-2 → ${STAGED_1_2_DIR}"
