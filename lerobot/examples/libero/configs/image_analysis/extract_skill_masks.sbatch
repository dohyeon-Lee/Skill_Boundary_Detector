#!/usr/bin/env bash
#SBATCH --job-name=skill_masks
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
DATADIR=libero_dataset           # libero_dataset, libero_small_dataset
IMAGE_KEY=observation.images.image  # observation.images.image, observation.images.wrist_image

SAM2_CHECKPOINT=${HOMEDIR}${PROJDIR}/models/sam2/sam2.1_hiera_large.pt
SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml   # sam2_hiera_t.yaml, sam2_hiera_s.yaml, sam2_hiera_b+.yaml, sam2_hiera_l.yaml, sam2.1_hiera_s.yaml

MAX_EPISODES_PER_TASK=1   # task당 처리할 에피소드 수. 0이면 전체
MIN_CENTROID_DIST=20.0    # centroid 이동 px 이상이면 changed
MIN_VISIBLE_RATIO=0.1     # end/start 면적 비율 이 미만이면 가려진 것으로 간주
DEPTH_CHECKPOINT=${HOMEDIR}${PROJDIR}/models/depth_anything/depth_anything_v2_small_hf
DEPTH_BG_MARGIN=20        # 평균 depth보다 이 % 초과하는 물체는 배경으로 간주, approach 제외
MIN_APPROACH_AREA=0.01   # approach 대상 object 최소 면적 비율 (이미지 대비)
DEPTH_WEIGHT=0.3          # 3D 거리 계산 시 Z(depth) 기여 가중치 (1.0=완전 3D, 0.0=순수 2D)
APPROACH_TIE_EPS=0.05         # 동점 판정 eps. relative모드: 0.05(=5%차이), absolute모드: 5.0(px)
TOP_CROP_RATIO=0.4            # 이미지 상단 이 비율 안에 centroid 있는 물체는 approach 제외 (0.0=비활성화)
CHANGED_PROP_BACK=2           # changed object 이전 방향 전파 스킬 수
CHANGED_PROP_FWD=1            # changed object 이후 방향 전파 스킬 수
APPROACH_SCORE_RELATIVE="--approach_score_relative"      # 비율 모드 (Δ/dist_start)
# APPROACH_SCORE_RELATIVE="--no-approach_score_relative"  # 절댓값 px 모드
WANDB_RUN_NAME=${DATA}_sam2_masks_ep${MAX_EPISODES_PER_TASK}

SKILLSET=${DATA}_skillset
SKILLSET_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET}/skills
DATASET_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_masks
# ─────────────────────────────────────────────────────────────────────────────

echo "Skillset : ${SKILLSET_DIR}"
echo "Dataset  : ${DATASET_DIR}"
echo "Output   : ${OUTPUT_DIR}"
echo "SAM2     : ${SAM2_CHECKPOINT}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/extract_skill_masks.py \
    --skillset_dir    ${SKILLSET_DIR} \
    --dataset_dir     ${DATASET_DIR} \
    --sam2_checkpoint ${SAM2_CHECKPOINT} \
    --sam2_config     ${SAM2_CONFIG} \
    --output_dir      ${OUTPUT_DIR} \
    --image_key       ${IMAGE_KEY} \
    --device                 cuda \
    --points_per_side        50 \
    --max_mask_area_ratio    0.35 \
    --min_mask_area_ratio    0.001 \
    --min_stability_score    0.85 \
    --max_episodes_per_task  ${MAX_EPISODES_PER_TASK} \
    --min_centroid_dist      ${MIN_CENTROID_DIST} \
    --min_visible_ratio      ${MIN_VISIBLE_RATIO} \
    --depth_checkpoint       ${DEPTH_CHECKPOINT} \
    --depth_bg_margin        ${DEPTH_BG_MARGIN} \
    --min_approach_area_ratio ${MIN_APPROACH_AREA} \
    --depth_weight            ${DEPTH_WEIGHT} \
    --approach_tie_eps        ${APPROACH_TIE_EPS} \
    --top_crop_ratio          ${TOP_CROP_RATIO} \
    --changed_prop_back       ${CHANGED_PROP_BACK} \
    --changed_prop_fwd        ${CHANGED_PROP_FWD} \
    ${APPROACH_SCORE_RELATIVE} \
    --gripper_point          128 45 \
    --wandb_project          image_analysis \
    --wandb_run_name         ${WANDB_RUN_NAME} \
    --wandb_log_every        1

echo "DONE → ${OUTPUT_DIR}"
