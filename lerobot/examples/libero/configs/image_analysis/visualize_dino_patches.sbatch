#!/usr/bin/env bash
#SBATCH --job-name=dino_patches
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --qos=big_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate
unset LD_LIBRARY_PATH

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA=libero_90
DATADIR=libero_dataset
IMAGE_KEY=observation.images.image

SAM2_CHECKPOINT=${HOMEDIR}${PROJDIR}/models/sam2/sam2.1_hiera_large.pt
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml

MAX_EPISODES_PER_TASK=1

MIN_CENTROID_DIST=20.0
MIN_VISIBLE_RATIO=0.1
TOP_CROP_RATIO=0.4
CHANGED_PROP_BACK=2

DINO_MODEL=${HOMEDIR}${PROJDIR}/models/dinov3-vits16   # patch_size=16 → 14x14=196 patches
PATCH_HIT_THRESHOLD=0.1   # 패치 내 마스크 픽셀 비율 threshold (0.0=any, 0.1=10%이상)

SKILLSET=${DATA}_skillset
SKILLSET_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET}/skills
DATASET_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}

WANDB_RUN_NAME=${DATA}_dino_patches_ep${MAX_EPISODES_PER_TASK}
# ─────────────────────────────────────────────────────────────────────────────

echo "Skillset : ${SKILLSET_DIR}"
echo "Dataset  : ${DATASET_DIR}"
echo "DINOv3   : ${DINO_MODEL}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/visualize_dino_patches.py \
    --skillset_dir           ${SKILLSET_DIR} \
    --dataset_dir            ${DATASET_DIR} \
    --sam2_checkpoint        ${SAM2_CHECKPOINT} \
    --sam2_config            ${SAM2_CONFIG} \
    --image_key              ${IMAGE_KEY} \
    --device                 cuda \
    --points_per_side        50 \
    --max_mask_area_ratio    0.35 \
    --min_mask_area_ratio    0.001 \
    --min_stability_score    0.85 \
    --gripper_point          128 45 \
    --max_episodes_per_task  ${MAX_EPISODES_PER_TASK} \
    --min_centroid_dist      ${MIN_CENTROID_DIST} \
    --min_visible_ratio      ${MIN_VISIBLE_RATIO} \
    --top_crop_ratio         ${TOP_CROP_RATIO} \
    --changed_prop_back      ${CHANGED_PROP_BACK} \
    --dino_model             ${DINO_MODEL} \
    --image_size             224 \
    --patch_size             16 \
    --patch_hit_threshold    ${PATCH_HIT_THRESHOLD} \
    --wandb_project          image_analysis \
    --wandb_run_name         ${WANDB_RUN_NAME} \
    --wandb_log_every        1

echo "DONE"
