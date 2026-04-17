#!/usr/bin/env bash
#SBATCH --job-name=build_skill_dataset
#SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

cd /scratch/mdorazi/Skill_Boundary_Detector/lerobot

source /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/activate

export WANDB_CACHE_DIR=/scratch/mdorazi/.wandb_cache
export WANDB_DATA_DIR=/scratch/mdorazi/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# ── 설정 ──────────────────────────────────────────────────────────────────────
DN_STEP=7
N_GMM=5

SMOOTH_WINDOW=7
SAVGOL_POLYORDER=4

REPLAN_INTERVAL=3
NMS_DIST=10

POLICY=dp_libero90_yonsei_pretrain
CHECKPOINT=080000
DATASET=libero_90

POLICY_PATH=/scratch/mdorazi/Skill_Boundary_Detector/outputs/${POLICY}/checkpoints/${CHECKPOINT}/pretrained_model
DATASET_DIR=/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/${DATASET}
OUTPUT_DIR=/scratch/mdorazi/Skill_Boundary_Detector/outputs/skill_dataset

WANDB_PROJECT=Skill_dataset
# ─────────────────────────────────────────────────────────────────────────────

# 각 array job이 5개 task 처리 (0-14 × 5 = task 0~74, 72 초과는 skip)
TASKS_PER_JOB=5
START_TASK=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))

TASK_IDS=""
for TASK_ID in $(seq ${START_TASK} ${END_TASK}); do
    if [ ${TASK_ID} -gt 72 ]; then
        break
    fi
    TASK_IDS="${TASK_IDS} ${TASK_ID}"
done

echo "Processing tasks:${TASK_IDS}"

WANDB_RUN_NAME=array${SLURM_ARRAY_TASK_ID}_tasks${START_TASK}-${END_TASK}

WANDB_MODE=online \
WANDB_DIR=${OUTPUT_DIR} \
PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  python examples/libero/build_skill_dataset.py \
    --dataset_dir ${DATASET_DIR} \
    --policy_path ${POLICY_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --task_ids ${TASK_IDS} \
    --replan_interval ${REPLAN_INTERVAL} \
    --nms_dist ${NMS_DIST} \
    --noise_scheduler_type DDIM \
    --num_inference_steps 10 \
    --eval_at_step ${DN_STEP} \
    --n_gmm_components ${N_GMM} \
    --smooth_window ${SMOOTH_WINDOW} \
    --savgol_polyorder ${SAVGOL_POLYORDER} \
    --resume \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "DONE array ${SLURM_ARRAY_TASK_ID}"
