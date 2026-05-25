#!/usr/bin/env bash
#SBATCH --job-name=pi05_FT

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node200
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
[ ! -f "${CONFIG_PY}" ] && CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
eval "$(python3 "${CONFIG_PY}" --shell | grep -E '^export (HOMEDIR|PROJDIR)=')"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

FT_DATASET=libero_10_op2_1
FT_DATASET_LOC=libero_small_dataset

# FT_DATASET=libero_10
# FT_DATASET_LOC=libero_dataset

PT_DATASET=libero_90
PT_check=070000
PT_BATCH_SIZE=16
PT_EXP=${PT_EXP:-}
EXP=${EXP:-}
LR=2.5e-05        # default: 2.5e-05
BATCH_SIZE=32
NUM_WORKERS=4     # shm 에러 시 0으로 낮출 것

VLA_OUTPUT_ROOT=${HOMEDIR}${PROJDIR}/VLA_outputs
PT_RUN_NAME=PT_${PT_DATASET}_pi05_batch${PT_BATCH_SIZE}
if [ -n "${PT_EXP}" ]; then
    PT_RUN_NAME=${PT_RUN_NAME}_${PT_EXP}
fi
RUN_NAME=FT_${PT_check}PT_${FT_DATASET}_pi05_batch${BATCH_SIZE}
if [ -n "${EXP}" ]; then
    RUN_NAME=${RUN_NAME}_${EXP}
fi
OUTPUT_DIR=${VLA_OUTPUT_ROOT}/${RUN_NAME}
CONFIG_PATH=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json
PT_MODEL_PATH=${VLA_OUTPUT_ROOT}/${PT_RUN_NAME}/checkpoints/${PT_check}/pretrained_model
RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
else
    echo "No checkpoint found. Starting fresh training."
fi

PYTORCH_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=lerobot/${FT_DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/${FT_DATASET_LOC}/${FT_DATASET} \
    --policy.type=pi05 \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${RUN_NAME} \
    --policy.pretrained_path=${PT_MODEL_PATH} \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_Finetune \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.push_to_hub=false \
    --save_freq=5000 \
    --steps=100000 \
    --policy.device=cuda \
    --dataset.video_backend=pyav \
    --batch_size=${BATCH_SIZE} \
    --policy.optimizer_lr=${LR} \
    --num_workers=${NUM_WORKERS} \
    ${RESUME_ARGS}

echo "DONE"
