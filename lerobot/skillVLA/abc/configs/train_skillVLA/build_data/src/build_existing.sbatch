#!/usr/bin/env bash
# Encode and assemble sequentially inside the existing allocation's worker.
set -euo pipefail

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SLURM_SUBMIT_DIR="${BUILD_DIR}"
export ABC_SKILLVLA_CONFIG="${ABC_SKILLVLA_CONFIG:-${BUILD_DIR}/train_skillVLA_config.yaml}"
BOOTSTRAP_PYTHON="${BUILD_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${BUILD_DIR}/src/train_skillVLA_config.py" \
  --config "${ABC_SKILLVLA_CONFIG}" --shell)"

if [ -f "${SKILL_LATENTS_PATH}" ]; then
  echo "[1/2] Encoded skill latents already exist -> skip"
else
  echo "[1/2] Encode FSQ epoch ${FSQ_CHECKPOINT}"
  bash "${BUILD_DIR}/src/encode_skills.sbatch"
fi

echo "[2/2] Build native annotations/ISS + spatial targets"
bash "${BUILD_DIR}/src/build_skillvla.sbatch"
