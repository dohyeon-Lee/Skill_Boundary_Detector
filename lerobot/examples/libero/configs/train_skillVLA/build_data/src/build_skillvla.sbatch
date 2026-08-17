#!/usr/bin/env bash
# Step 5 (final): assemble the SkillVLA dataset and final outputs.
#   (b) add_skill_latents_to_dataset → skillvla/ (skill columns) + skill_initial_state.npz (ISS)
#   (c) copy FSQ checkpoint → FSQ.pt
#   (d) optional cleanup of intermediates (_work, skill_latents.npz)
# Note: DINO precompute 완전 제거 — terminator co-training은 배치 현재 프레임을 ONLINE 토큰화. 이전엔
# kept only for build_data_eval. The per-frame observation.dino.image train cache is removed.
#SBATCH --job-name=skillVLA_build
#SBATCH --gres=gpu:1
#SBATCH --requeue
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

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA BUILD"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "Source   : ${RAW_DATASET_DIR}"
echo "Latents  : ${SKILL_LATENTS_PATH}"
echo "ISS npz  : ${ISS_NPZ_PATH}   (pmax=${SKILL_PMAX}, distribution=${SKILL_JITTER_DISTRIBUTION})"
echo "FSQ.pt   : ${FSQ_COPY_PATH}"
echo "skillvla : ${SKILLVLA_DATASET_DIR}"

mkdir -p "${SKILLVLA_RUN_DIR}"

# (DINO precompute 단계 (a)/(a') 은퇴 — dino.npz/dino_wrist.npz 산출물 없음.)

# ── (b) add skill columns → SkillVLA LeRobot dataset + skill_initial_state.npz ──
read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/add_skill_latents_to_dataset.py \
    --src_dataset_dir "${RAW_DATASET_DIR}" \
    --dst_dataset_dir "${SKILLVLA_DATASET_DIR}" \
    --dst_repo_id     "skillvla/${SOURCE_DATASET}_${RUN_TAG}" \
    --latents_path    "${SKILL_LATENTS_PATH}" \
    --iss_npz_path    "${ISS_NPZ_PATH}" \
    --pmax            "${SKILL_PMAX}" \
    --jitter_distribution "${SKILL_JITTER_DISTRIBUTION}" \
    --fsq_levels      "${FSQ_LEVELS[@]}" \
    --max_order       "${MAX_ORDER}" \
    --max_length      "${MAX_LENGTH}"

# ── (c) copy the FSQ checkpoint used ──
cp -f "${FSQ_MODEL_PATH}" "${FSQ_COPY_PATH}"
echo "Copied FSQ checkpoint → ${FSQ_COPY_PATH}"

# ── (d) transition pack — stage-3 / FT-replay training set (segment-level skill-start pairs).
# Built here so every fresh dataset ships with it; rebuild old packs that lack the raw start-state window.
TRANSITION_PACK="${SKILLVLA_RUN_DIR}/transitions.npz"
if [ -f "${TRANSITION_PACK}" ] && "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import numpy as np,sys; z=np.load(sys.argv[1]); assert "start_state" in z and "jitter_distribution" in z and int(z["schema_version"]) >= 3' \
    "${TRANSITION_PACK}"; then
  echo "Transition pack schema OK: ${TRANSITION_PACK}"
else
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/build_transition_pack.py" \
    --skillvla_dir="${SKILLVLA_RUN_DIR}/skillvla" \
    || echo "transition-pack build failed (non-fatal — stage3 lazy-builds it)"
fi

# ── (e) cleanup intermediates ──
# Only this DP's segmentation (skillset + skill_tokens) and this run's latents.
# The shared dino/ stays so other DP/FSQ runs on the same source can reuse it.
if [ "${CLEANUP_INTERMEDIATE}" = "true" ]; then
  echo "Cleaning up intermediates ..."
  rm -rf "${SKILLVLA_SEG_DIR}"
  rm -f "${SKILL_LATENTS_PATH}"
  echo "  removed ${SKILLVLA_SEG_DIR} and ${SKILL_LATENTS_PATH}"
fi

echo "DONE → ${SKILLVLA_RUN_DIR}  (skillvla/, skill_initial_state.npz, FSQ.pt)"
