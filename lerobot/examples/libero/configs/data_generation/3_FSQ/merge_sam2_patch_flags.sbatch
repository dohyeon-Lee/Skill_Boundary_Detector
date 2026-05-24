#!/usr/bin/env bash
# Merge per-skill SAM2 mask files into one FSQ patch-flag npz (CPU job).
# All values are injected by run_dino_parallel.sh.
#SBATCH --job-name=merge_SAM2_flags
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
PATCH_GRID=${PATCH_GRID:-8}
DERIVED_DATA_DIR=${DERIVED_DATA_DIR:-${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data}
FSQ_PRECOMPUTE_DIR=${FSQ_PRECOMPUTE_DIR:-${DERIVED_DATA_DIR}/${DATA}_for_FSQ}
SKILLSET_DIR=${SKILLSET_DIR:-${FSQ_PRECOMPUTE_DIR}/${SKILLSET}}
SAM2_OUTPUT_DIR=${SAM2_OUTPUT_DIR:-${FSQ_PRECOMPUTE_DIR}/sam2_masks}
SAM2_MERGED_PATH=${SAM2_MERGED_PATH:-${FSQ_PRECOMPUTE_DIR}/patch_flags.npz}
mkdir -p "$(dirname "${SAM2_MERGED_PATH}")"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src:${HOMEDIR}${PROJDIR}/lerobot/examples/libero \
  python -u examples/libero/merge_sam2_patch_flags.py \
    --skills_dir     ${SKILLSET_DIR}/skills \
    --sam2_masks_dir ${SAM2_OUTPUT_DIR} \
    --output_path    ${SAM2_MERGED_PATH} \
    --n_patches      $((PATCH_GRID * PATCH_GRID))

echo "DONE -> ${SAM2_MERGED_PATH}"
