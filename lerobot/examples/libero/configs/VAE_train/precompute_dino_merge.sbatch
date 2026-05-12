#!/usr/bin/env bash
# Merge partial DINO feature files — all server-specific values injected via run_dino_parallel.sh
#SBATCH --job-name=DINO_merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

: "${HOMEDIR:?}" "${PROJDIR:?}" "${N_WORKERS:?}"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

echo "START DINO MERGE  |  N_WORKERS=${N_WORKERS}"

DATA=${DATA:-libero_90}
DATADIR=${DATADIR:-libero_dataset}
SKILLSET=${DATA}_skillset
VISUAL_BACKBONE=${VISUAL_BACKBONE:-dinov2_small}

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/precompute_dino_merge.py \
    --skills_dir    ${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET} \
    --dataset_dir   ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA} \
    --output_path   ${IMAGE_FEATURES_PATH} \
    --image_model_name ${IMAGE_MODEL_PATH} \
    --num_workers   ${N_WORKERS}

echo "DONE ${IMAGE_FEATURES_PATH}"
