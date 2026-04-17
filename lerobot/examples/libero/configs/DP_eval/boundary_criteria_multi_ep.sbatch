#!/usr/bin/env bash
#SBATCH --job-name=vf_error_multi
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
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

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
TASK_ID=8
N_EPISODES=5   
N_GMM=5
DN_STEP=8

# GMM 개수, SG 필터, mean 필터 정도 실험해보고 결정하면될듯. cos diversity로 재는게 맞는 것 같고.

POLICY_PATH=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model
DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/vf_error_libero90/task${TASK_ID}_multi


PLOT_GMM_3D=true   # true로 바꾸면 에피소드별 인터랙티브 3D HTML 생성 (시간 추가 소요)

WANDB_PROJECT=DP_eval
WANDB_RUN_NAME=task${TASK_ID}_${N_EPISODES}eps_${N_GMM}gmm_${DN_STEP}step
# ─────────────────────────────────────────────────────────────────────────────

# 현재 GMM 클러스터 mean hard assignment로 계산됨 (mean_full[:,:3])
# soft EM이 이론상 더 나을 가능성 있음. 향후 soft EM으로 개선 고려.

# GMM cluster num과 cos diversity사이 상관관계 : N클수록 cos div값의 outlier가 줄어드는 경향 보임 8이 젤 괜찮은듯

echo "=== GMM n_components=${N_GMM} ==="
PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/replay_demo.py \
    --dataset_dir ${DATASET_DIR} \
    --task_id ${TASK_ID} \
    --n_episodes ${N_EPISODES} \
    --policy_path ${POLICY_PATH} \
    --output_dir ${OUTPUT_DIR}_gmm${N_GMM} \
    --replan_interval 5 \
    --mse_smooth_method savgol \
    --mse_smooth_window 6 \
    --savgol_polyorder 4 \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step ${DN_STEP} \
    --n_gmm_components ${N_GMM} \
    $([ "${PLOT_GMM_3D}" = "true" ] && echo "--plot_gmm_3d") \
    --no-save_skills \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}


echo "DONE"
