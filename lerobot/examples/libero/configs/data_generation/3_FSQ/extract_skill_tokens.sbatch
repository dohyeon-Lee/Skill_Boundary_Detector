#!/usr/bin/env bash
# Extract skill-level DINO tokens from precomputed frame DINO features.
# CPU-only; all paths are injected by run_dino_parallel.sh.
#SBATCH --job-name=skill_tokens
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
SKILLSET=${SKILLSET:-${DATA}_skillset}
VISUAL_BACKBONE=${VISUAL_BACKBONE:-dinov3_vits16}
PATCH_GRID=${PATCH_GRID:-8}
DERIVED_DATA_DIR=${DERIVED_DATA_DIR:-${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data}
FSQ_PRECOMPUTE_DIR=${FSQ_PRECOMPUTE_DIR:-${DERIVED_DATA_DIR}/${DATA}_for_FSQ}
SKILLSET_DIR=${SKILLSET_DIR:-${FSQ_PRECOMPUTE_DIR}/${SKILLSET}}

source examples/libero/configs/data_generation/3_FSQ/visual_backbone.sh
resolve_visual_backbone

FRAME_DINO_DIR="${DERIVED_DATA_DIR}/${DATA}_DINO/${IMAGE_FEATURE_TAG}_pg${PATCH_GRID}"
DINO_TOKENS_PATH=${DINO_TOKENS_PATH:-${FSQ_PRECOMPUTE_DIR}/${IMAGE_FEATURE_TAG}_tokens.npz}
mkdir -p "$(dirname "${DINO_TOKENS_PATH}")"

echo "Frame DINO dir : ${FRAME_DINO_DIR}"
echo "Output tokens  : ${DINO_TOKENS_PATH}"

if [ ! -d "${FRAME_DINO_DIR}" ]; then
  echo "ERROR: Frame DINO directory not found: ${FRAME_DINO_DIR}" >&2
  echo "Run data_generation/1_DINO/run_frame_dino_parallel.sh first." >&2
  exit 1
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "ERROR: skillset directory not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run data_generation/2_skillset/build_skill_dataset.sbatch first." >&2
  exit 1
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/extract_skill_dino_tokens.py \
    --frame_dino_dir  ${FRAME_DINO_DIR} \
    --skills_dir      ${SKILLSET_DIR}/skills \
    --output_path     ${DINO_TOKENS_PATH} \
    --image_key       observation.images.image

echo "DONE → ${DINO_TOKENS_PATH}"
