#!/usr/bin/env bash
#SBATCH --job-name=DP_eval

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# HOMEDIR=/scratch/mdorazi
# PROJDIR=/Skill_Boundary_Detector

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 스킬 영상 상단에 episode번호, 해당 episode에서 몇번째 스킬인지도 표시 필요
DATASET=libero_90
VISUAL_BACKBONE=dinov3_vits16
PATCH_GRID=8
DP=dp_${DATASET}_dino_${VISUAL_BACKBONE}_pg${PATCH_GRID}_obs10_horizon16
DINO_FEATURE_DIR=/data2/dohyeon/SBD/libero_dataset/${DATASET}_data/${DATASET}_DINO/${VISUAL_BACKBONE}_pg${PATCH_GRID}
CHECKPOINT=060000
TASK_ID=2
N_EPISODES=5
N_GMM=5
DN_STEP=7

SMOOTH_WINDOW=7 
SAVGOL_POLYORDER=4

REPLAN_INTERVAL=3 
NMS_DIST=25

# POLICY_PATH=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/060000/pretrained_model # 5 16
POLICY_PATH=${HOMEDIR}${PROJDIR}/outputs/${DP}/checkpoints/${CHECKPOINT}/pretrained_model # 10 16
DATASET_DIR=${HOMEDIR}${PROJDIR}/libero_dataset/${DATASET}
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/skill_divider_${DATASET}/task${TASK_ID}

PLOT_GMM_3D=false   # true로 바꾸면 에피소드별 인터랙티브 3D HTML 생성
PLOT_PRED_MSE=true # true로 바꾸면 boundary criteria 그래프에 pred-demo xyz MSE subplot 추가
MSE_WINDOW=4        # pred-demo MSE 계산에 사용할 chunk 앞 N 스텝

WANDB_PROJECT=DP_eval
WANDB_RUN_NAME=task${TASK_ID}_${N_EPISODES}eps_${N_GMM}gmm
# ─────────────────────────────────────────────────────────────────────────────

# WANDB_MODE=online \
# WANDB_DIR=${OUTPUT_DIR} \
PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/skill_divider.py \
    --dataset_dir ${DATASET_DIR} \
    --task_id ${TASK_ID} \
    --n_episodes ${N_EPISODES} \
    --policy_path ${POLICY_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --replan_interval ${REPLAN_INTERVAL} \
    --nms_dist ${NMS_DIST} \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step ${DN_STEP} \
    --n_gmm_components ${N_GMM} \
    --smooth_window ${SMOOTH_WINDOW} \
    --savgol_polyorder ${SAVGOL_POLYORDER} \
    $([ "${PLOT_GMM_3D}" = "true" ] && echo "--plot_gmm_3d") \
    $([ "${PLOT_PRED_MSE}" = "true" ] && echo "--plot_pred_mse --mse_window ${MSE_WINDOW}") \
    --dino_feature_dir ${DINO_FEATURE_DIR} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "DONE"
