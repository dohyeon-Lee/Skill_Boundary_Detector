#!/usr/bin/env bash
# Worker template — all server-specific values injected via run_dino_parallel.sh
#SBATCH --job-name=DINO_worker
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

: "${HOMEDIR:?}" "${PROJDIR:?}" "${N_WORKERS:?}"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

echo "Node: $SLURMD_NODENAME  |  Worker: ${SLURM_ARRAY_TASK_ID} / ${N_WORKERS}"
nvidia-smi | head -20

DATA=${DATA:-libero_90}
DATADIR=${DATADIR:-libero_dataset}
SKILLSET=${DATA}_skillset
VISUAL_BACKBONE=${VISUAL_BACKBONE:-dinov2_small}

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/precompute_dino_features.py \
    --skills_dir       ${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET} \
    --dataset_dir      ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA} \
    --output_path      ${DINO_TOKENS_PATH} \
    --image_model_name ${IMAGE_MODEL_PATH} \
    --image_size       224 \
    --patch_grid       8 \
    --n_patch_raw      196 \
    --batch_size       256 \
    --device           cuda \
    --num_workers      ${N_WORKERS} \
    --worker_id        ${SLURM_ARRAY_TASK_ID}

echo "DONE worker ${SLURM_ARRAY_TASK_ID}"
