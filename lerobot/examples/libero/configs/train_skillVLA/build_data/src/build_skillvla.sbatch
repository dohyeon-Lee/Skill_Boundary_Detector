#!/usr/bin/env bash
# Step 5 (final): assemble the SkillVLA dataset and final outputs.
#   (a) merge per-episode DINO → dino.npz        (for build_data_eval viz only; merge/copy, cheap)
#   (b) add_skill_latents_to_dataset → skillvla/ (skill columns) + skill_initial_state.npz (ISS)
#   (c) copy FSQ checkpoint → FSQ.pt
#   (d) optional cleanup of intermediates (_work, skill_latents.npz)
# Note: Stage-2 training does NOT use dino.npz (VLM/expert encode raw images on the fly); it is
# kept only for build_data_eval. The per-frame observation.dino.image train cache is removed.
#SBATCH --job-name=skillVLA_build
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

echo "START SKILLVLA BUILD"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "Source   : ${RAW_DATASET_DIR}"
echo "Latents  : ${SKILL_LATENTS_PATH}"
echo "dino.npz : ${DINO_NPZ_PATH}   (eval-only)"
echo "ISS npz  : ${ISS_NPZ_PATH}   (pmax=${SKILL_PMAX})"
echo "FSQ.pt   : ${FSQ_COPY_PATH}"
echo "skillvla : ${SKILLVLA_DATASET_DIR}"

mkdir -p "${SKILLVLA_RUN_DIR}"

# ── (a) merge per-episode DINO → dino.npz (build_data_eval viz only; not used by training) ──
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/merge_frame_dino.py" \
    --frame_dino_dir "${FSQ_DINO_DIR}" \
    --image_key      "${DINO_IMAGE_KEY}" \
    --output_path    "${DINO_NPZ_PATH}"

# ── (a') wrist DINO → dino_wrist.npz (only when dino_wrist:true; Stage-1 terminator_use_wrist tokens).
# Same merge, wrist camera. Reuses the already-precomputed wrist features in FSQ_DINO_DIR (no GPU). ──
if [ -n "${DINO_WRIST_NPZ_PATH:-}" ]; then
  echo "dino_wrist : ${DINO_WRIST_NPZ_PATH}   (terminator wrist tokens)"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/merge_frame_dino.py" \
      --frame_dino_dir "${FSQ_DINO_DIR}" \
      --image_key      "${WRIST_IMAGE_KEY}" \
      --output_path    "${DINO_WRIST_NPZ_PATH}"
fi

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
    --fsq_levels      "${FSQ_LEVELS[@]}" \
    --max_order       "${MAX_ORDER}" \
    --max_length      "${MAX_LENGTH}"

# ── (c) copy the FSQ checkpoint used ──
cp -f "${FSQ_MODEL_PATH}" "${FSQ_COPY_PATH}"
echo "Copied FSQ checkpoint → ${FSQ_COPY_PATH}"

# ── (d) cleanup intermediates ──
# Only this DP's segmentation (skillset + skill_tokens) and this run's latents.
# The shared dino/ stays so other DP/FSQ runs on the same source can reuse it.
if [ "${CLEANUP_INTERMEDIATE}" = "true" ]; then
  echo "Cleaning up intermediates ..."
  rm -rf "${SKILLVLA_SEG_DIR}"
  rm -f "${SKILL_LATENTS_PATH}"
  echo "  removed ${SKILLVLA_SEG_DIR} and ${SKILL_LATENTS_PATH}"
fi

echo "DONE → ${SKILLVLA_RUN_DIR}  (skillvla/, skill_initial_state.npz, dino.npz, FSQ.pt)"
