#!/usr/bin/env bash
#SBATCH --job-name=VAE_train

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
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

DATA=libero_90
LATENT_DIM=64
VAE_TYPE=spline   # 'raw' (BiLSTM) or 'spline' (MLP)
# SKILLSET=${DATA}_skillset_fixed40
SKILLSET=${DATA}_skillset


EXP_NAME="${DATA}_lat${LATENT_DIM}_${VAE_TYPE}"
OUTPUT_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

# checkpoint filename differs per vae_type
if [ "${VAE_TYPE}" = "spline" ]; then
  CKPT_PATTERN="spline_vae_epoch*.pt"
else
  CKPT_PATTERN="skill_vae_epoch*.pt"
fi

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
  LATEST_CKPT=$(ls -v "${OUTPUT_DIR}"/${CKPT_PATTERN} 2>/dev/null | tail -1 || true)
  if [ -n "${LATEST_CKPT}" ]; then
    RESUME_ARG="--resume_from ${LATEST_CKPT}"
    echo "Resuming from ${LATEST_CKPT}"
  else
    echo "No checkpoint found, training from scratch"
  fi
else
  echo "No checkpoint found, training from scratch"
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/train_vae.py \
    --skills_dir ${HOMEDIR}${PROJDIR}/libero_dataset/${SKILLSET} \
    --output_dir ${OUTPUT_DIR} \
    --vae_type ${VAE_TYPE} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim 128 \
    --num_layers 2 \
    --epochs 10000 \
    --beta 0.01 \
    --lr 1.5e-3 \
    --batch_size 512 \
    --checkpoint_every 1000 \
    --wandb_project VAE_train \
    --wandb_run_name ${EXP_NAME} \
    ${RESUME_ARG}
#     --beta 0.001 \
echo "DONE ${EXP_NAME}"


##3e-4 \
