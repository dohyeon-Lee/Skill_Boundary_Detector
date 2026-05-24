#!/usr/bin/env bash
# SAM2 worker for FSQ patch flags.
# All dataset/server values are injected by run_dino_parallel.sh.
#SBATCH --job-name=SAM2_masks
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
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
SKILLSET=${SKILLSET:-${DATA}_skillset}
DERIVED_DATA_DIR=${DERIVED_DATA_DIR:-${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data}
FSQ_PRECOMPUTE_DIR=${FSQ_PRECOMPUTE_DIR:-${DERIVED_DATA_DIR}/${DATA}_for_FSQ}
SKILLSET_DIR=${SKILLSET_DIR:-${FSQ_PRECOMPUTE_DIR}/${SKILLSET}}

SAM2_CHECKPOINT=${SAM2_CHECKPOINT:-${HOMEDIR}${PROJDIR}/models/sam2/sam2.1_hiera_large.pt}
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml

if [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "ERROR: SAM2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python -u examples/libero/precompute_sam2_masks.py \
    --skillset_dir           ${SKILLSET_DIR}/skills \
    --dataset_dir            ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA} \
    --output_dir             ${SAM2_OUTPUT_DIR:-${FSQ_PRECOMPUTE_DIR}/sam2_masks} \
    --sam2_checkpoint        ${SAM2_CHECKPOINT} \
    --sam2_config            ${SAM2_CONFIG} \
    --image_key              observation.images.image \
    --device                 cuda \
    --points_per_side        50 \
    --max_mask_area_ratio    0.35 \
    --min_mask_area_ratio    0.001 \
    --min_stability_score    0.85 \
    --gripper_point          128 45 \
    --min_centroid_dist      30.0 \
    --min_visible_ratio      0.05 \
    --changed_top_crop_ratio 0.35 \
    --green_top_crop_ratio   0.35 \
    --changed_prop_back      2 \
    --image_size             224 \
    --patch_size             16 \
    --patch_hit_threshold    0.1 \
    --pool_size              8 \
    --num_workers            ${N_WORKERS} \
    --worker_id              ${SLURM_ARRAY_TASK_ID}

echo "DONE worker ${SLURM_ARRAY_TASK_ID}"
