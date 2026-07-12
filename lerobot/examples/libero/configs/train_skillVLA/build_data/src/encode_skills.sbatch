#!/usr/bin/env bash
# Step 4 of SkillVLA data generation:
#   (a) extract skill-level DINO tokens from the per-episode DINO + skillset
#   (b) encode each skill with the trained FSQ → skill latent vectors
#
# Outputs:
#   {skillvla_dataset}/{source}/_work/seg_{dp}_ck{ckpt}/skill_tokens.npz   (DINO tokens, FSQ-independent)
#   {skillvla_dataset}/{source}/{run_tag}/skill_latents.npz   (FSQ vectors)
#SBATCH --job-name=skillVLA_encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # train_skillVLA
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA ENCODE"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "Skillset : ${SKILLSET_DIR}/skills"
echo "DINO dir : ${FSQ_DINO_DIR}"
echo "FSQ      : ${FSQ_MODEL_PATH}"
echo "Tokens   : ${SKILL_TOKENS_PATH}"
echo "Latents  : ${SKILL_LATENTS_PATH}"
nvidia-smi || true

if [ ! -f "${FSQ_MODEL_PATH}" ]; then
  echo "FSQ model not found: ${FSQ_MODEL_PATH}" >&2
  exit 1
fi

# ── (a) extract skill-level DINO tokens (CPU). FSQ-independent → reuse if it already exists
#        (the same skillset/pg8 DINO tokens are shared across FSQ epochs; set FORCE_EXTRACT=1 to rebuild).
if [ "${FORCE_EXTRACT:-0}" != "1" ] && [ -f "${SKILL_TOKENS_PATH}" ]; then
  echo "skill_tokens.npz exists → reuse (FSQ-independent): ${SKILL_TOKENS_PATH}"
else
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/extract_skill_dino_tokens.py \
      --frame_dino_dir "${FSQ_DINO_DIR}" \
      --skills_dir     "${SKILLSET_DIR}/skills" \
      --output_path    "${SKILL_TOKENS_PATH}" \
      --image_key      "${DINO_IMAGE_KEY}"
fi

# ── (b) encode skills with FSQ → latent vectors (GPU) ──
# transfer 안전망(B): fsq_snap_to_supported:true면 미지원 코드를 최근접 지원 코드로 snap.
SNAP_ARGS=(--no-snap_to_supported)
if [ "${FSQ_SNAP_TO_SUPPORTED:-false}" = "true" ]; then
  SNAP_REF="${FSQ_SNAP_REFERENCE:-}"
  if [ -z "${SNAP_REF}" ]; then
    echo "fsq_snap_to_supported=true 인데 fsq_snap_reference 가 비어있음 (원본 skill_code_freq.npz 경로 필요)" >&2
    exit 1
  fi
  case "${SNAP_REF}" in /*) ;; *) SNAP_REF="${PROJECT_ROOT}/${SNAP_REF}" ;; esac
  SNAP_ARGS=(--snap_to_supported --supported_freq_path "${SNAP_REF}" --min_code_freq "${FSQ_SNAP_MIN_CODE_FREQ:-1}")
fi
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/encode_FSQ_skills.py \
    --model_path    "${FSQ_MODEL_PATH}" \
    --skills_dir    "${SKILLSET_DIR}/skills" \
    --output_path   "${SKILL_LATENTS_PATH}" \
    --device        cuda \
    "${SNAP_ARGS[@]}"

echo "DONE → ${SKILL_LATENTS_PATH}"
