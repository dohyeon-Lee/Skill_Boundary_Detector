#!/usr/bin/env bash
#SBATCH --job-name=verify_skills
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot

source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

# ── 설정 ──────────────────────────────────────────────────────────────────────
TASK_ID=0
EPISODE_ID=1

DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90
SKILL_DIVIDER_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/skill_divider/task${TASK_ID}
SKILL_DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/skill_dataset/skills/task$(printf "%02d" ${TASK_ID})
# ─────────────────────────────────────────────────────────────────────────────

PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/verify_skills.py \
    --task_id ${TASK_ID} \
    --episode_id ${EPISODE_ID} \
    --skill_divider_dir ${SKILL_DIVIDER_DIR} \
    --skill_dataset_dir ${SKILL_DATASET_DIR} \
    --dataset_dir ${DATASET_DIR}

echo "DONE"
