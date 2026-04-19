#!/usr/bin/env bash
#SBATCH --job-name=vf_error_90
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
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
TASK_ID=0
EPISODE_ID=2        # 분석할 에피소드 번호

POLICY_PATH=/scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model
DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/vf_error_libero90/task${TASK_ID}_ep${EPISODE_ID}

WANDB_PROJECT=DP_eval
WANDB_RUN_NAME=vf_error_task${TASK_ID}_ep${EPISODE_ID}
# ─────────────────────────────────────────────────────────────────────────────

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/replay_demo.py \
    --dataset_dir ${DATASET_DIR} \
    --episode_ids "[${EPISODE_ID}]" \
    --policy_path ${POLICY_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --replan_interval 5 \
    --mse_smooth_method savgol \
    --mse_smooth_window 5 \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step 8 \
    --no-save_skills \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "DONE"
