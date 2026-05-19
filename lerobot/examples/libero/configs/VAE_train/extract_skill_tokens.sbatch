#!/usr/bin/env bash
# Extract skill-level DINO tokens from pre-computed frame features (CPU, no GPU needed).
# All values injected via run_dino_parallel.sh.
#SBATCH --job-name=skill_tokens
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

: "${HOMEDIR:?}" "${PROJDIR:?}"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

DATA=${DATA:-libero_90}
DATADIR=${DATADIR:-libero_dataset}
SKILLSET=${DATA}_skillset
VISUAL_BACKBONE=${VISUAL_BACKBONE:-dinov3_vits16}
PATCH_GRID=${PATCH_GRID:-8}

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

FRAME_DINO_DIR="${HOMEDIR}${PROJDIR}/outputs/${DATA}_frame_dino_features/${IMAGE_FEATURE_TAG}_pg${PATCH_GRID}"

echo "Frame DINO dir : ${FRAME_DINO_DIR}"
echo "Output tokens  : ${DINO_TOKENS_PATH}"

if [ ! -d "${FRAME_DINO_DIR}" ]; then
  echo "ERROR: Frame DINO directory not found: ${FRAME_DINO_DIR}" >&2
  echo "Run DP_train/run_frame_dino_parallel.sh first." >&2
  exit 1
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/extract_skill_dino_tokens.py \
    --frame_dino_dir  ${FRAME_DINO_DIR} \
    --skills_dir      ${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET}/skills \
    --output_path     ${DINO_TOKENS_PATH} \
    --image_key       observation.images.image

echo "DONE → ${DINO_TOKENS_PATH}"
