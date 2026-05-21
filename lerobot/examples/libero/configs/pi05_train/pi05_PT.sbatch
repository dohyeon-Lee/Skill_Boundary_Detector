#!/usr/bin/env bash
#SBATCH --job-name=pi05_PT

# for yonsei cluster: 
# suma_pro6000, asus_pro6000                            : ~96GB GPU memory, --qos=pro6000_qos
# suma_a6000, asus_6000ada, gigabyte_a6000, tyan_a6000  : ~48GB GPU memory, --qos=big_qos, base_qos
# model size : 
# BATCH_SIZE=32                                         : 48593MiB       
# BATCH_SIZE=16                                         : 40374MiB

# for RLLAB: 
# node100, node01                                       : 46068MiB

# SBATCH --partition=suma_a6000
# SBATCH --qos=base_qos

#SBATCH --partition=debug
# SBATCH --nodelist=node01
#SBATCH --qos=big_qos

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# HOMEDIR=/scratch/mdorazi
# PROJDIR=/Skill_Boundary_Detector

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

# -------------------------------------------------------------------------------- #
DATASET=libero_10_op1_10 # libero_90 / libero_10_op1_10
DATASET_ROOT=libero_small_dataset # libero_dataset / libero_small_dataset
BATCH_SIZE=16
NUM_GPUS=1      # --gres=gpu:N 과 맞춰야 함
NUM_WORKERS=4   # shm 에러 시 0 유지, 여유 있으면 2~4로 올릴 것
LR=$(python3 -c "print(2.5e-05 * ${NUM_GPUS})")
# -------------------------------------------------------------------------------- #

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# GPU가 완전히 빌 때까지 대기 (사용 중인 메모리가 500 MiB 이하일 때 시작)
GPU_FREE_THRESHOLD_MIB=500
echo "Waiting for GPU to be free (threshold: ${GPU_FREE_THRESHOLD_MIB} MiB)..."
while true; do
    USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "${USED_MIB}" -le "${GPU_FREE_THRESHOLD_MIB}" ]; then
        echo "GPU is free (${USED_MIB} MiB used). Starting training."
        break
    fi
    echo "  GPU in use: ${USED_MIB} MiB. Waiting 60s..."
    sleep 60
done



CONFIG_PATH=../outputs/pi05_${DATASET}/checkpoints/last/pretrained_model/train_config.json

OUTPUT_DIR=../outputs/pi05_${DATASET}_batch${BATCH_SIZE}
RESUME_ARGS=""
if [ -f "${CONFIG_PATH}" ]; then
    echo "Checkpoint found. Resuming from ${CONFIG_PATH}"
    RESUME_ARGS="--resume=true --config_path=${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint found. Removing existing output dir and starting fresh."
    rm -rf "${OUTPUT_DIR}"
else
    echo "No checkpoint found. Starting fresh training."
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=${NUM_GPUS} ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-train \
    --dataset.repo_id=lerobot/${DATASET} \
    --dataset.root=${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET} \
    --policy.type=pi05 \
    --output_dir=${OUTPUT_DIR} \
    --job_name=pi05_PT_${DATASET} \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=VLA_posttrain \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
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
