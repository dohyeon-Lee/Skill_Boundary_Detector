#!/usr/bin/env bash
#SBATCH --job-name=DP_skill

# SBATCH --partition=suma_a6000,asus_6000ada,asus_pro6000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=base_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}${PROJDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}${PROJDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

CONFIG_PY=examples/libero/configs/data_generation/pipeline_config.py
eval "$(python "${CONFIG_PY}" --shell)"

# ── Skill division hyperparameters ───────────────────────────────────────────
DN_STEP=7
N_GMM=5

SMOOTH_WINDOW=7
SAVGOL_POLYORDER=4

REPLAN_INTERVAL=3
NMS_DIST=25

DATASET=${DATA}
POLICY=${DP_POLICY}
CHECKPOINT=${DP_CHECKPOINT}
DINO_FEATURE_DIR=${DINO_FEATURE_DIR}

POLICY_PATH=${DP_POLICY_PATH}
DATASET_ROOT=${DATADIR}
DATASET_DIR=${RAW_DATASET_DIR}
OUTPUT_DIR=${SKILLSET_DIR}

WANDB_PROJECT=Skill_dataset
# ─────────────────────────────────────────────────────────────────────────────

# Each array job processes 5 tasks.
# libero_90: array=0-14, MAX_TASK_ID=72, TASKS_PER_JOB=5.
MAX_TASK_ID=72
TASKS_PER_JOB=5
START_TASK=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_JOB ))
END_TASK=$(( START_TASK + TASKS_PER_JOB - 1 ))

TASK_IDS=""
for TASK_ID in $(seq ${START_TASK} ${END_TASK}); do
    if [ ${TASK_ID} -gt ${MAX_TASK_ID} ]; then
        break
    fi
    TASK_IDS="${TASK_IDS} ${TASK_ID}"
done

echo "Processing tasks:${TASK_IDS}"

WANDB_RUN_NAME=array${SLURM_ARRAY_TASK_ID}_tasks${START_TASK}-${END_TASK}

WANDB_MODE=online \
WANDB_DIR=${OUTPUT_DIR} \

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
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
    --dino_feature_dir ${DINO_FEATURE_DIR} \
    --resume \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "DONE array ${SLURM_ARRAY_TASK_ID}"
