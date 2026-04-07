#!/usr/bin/env bash
#SBATCH --job-name=skill_generate_90
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot

source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# 각 array job이 5개 task 처리 (0-14 × 5 = task 0~74, 72 초과는 skip)
TASKS_PER_JOB=5
START_TASK=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))

for TASK_ID in $(seq ${START_TASK} ${END_TASK}); do
    if [ ${TASK_ID} -gt 72 ]; then
        break
    fi

    echo "--- Processing task ${TASK_ID} ---"

    PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
      python examples/libero/replay_demo.py \
        --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
        --task_id ${TASK_ID} \
        --n_episodes 60 \
        --policy_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model \
        --replan_interval 5 \
        --mse_smooth_method savgol \
        --mse_smooth_window 5 \
        --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero90/task${TASK_ID} \
        --save_skills \
        --skip_viz \
        --wandb_project SBD_skill_90

    echo "DONE task ${TASK_ID}"
done

echo "DONE array ${SLURM_ARRAY_TASK_ID}"
