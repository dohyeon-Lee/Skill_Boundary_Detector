#!/usr/bin/env bash
#SBATCH --job-name=skill_divider
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31,node25
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot

source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate


export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 스킬 영상 상단에 episode번호, 해당 episode에서 몇번째 스킬인지도 표시 필요
TASK_ID=1
N_EPISODES=3
N_GMM=5
DN_STEP=7

SMOOTH_WINDOW=7 
SAVGOL_POLYORDER=4

REPLAN_INTERVAL=3 
NMS_DIST=30

POLICY_PATH=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model
DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/skill_divider/task${TASK_ID}

PLOT_GMM_3D=false   # true로 바꾸면 에피소드별 인터랙티브 3D HTML 생성

WANDB_PROJECT=DP_eval
WANDB_RUN_NAME=task${TASK_ID}_${N_EPISODES}eps_${N_GMM}gmm_${DN_STEP}step
# ─────────────────────────────────────────────────────────────────────────────

# WANDB_MODE=online \
# WANDB_DIR=${OUTPUT_DIR} \
PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
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
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "DONE"
