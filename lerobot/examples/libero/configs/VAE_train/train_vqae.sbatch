#!/usr/bin/env bash
#SBATCH --job-name=VQAE_train

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
NUM_EMBEDDINGS=512
SKILLSET=${DATA}_skillset_fixed50
# SKILLSET=${DATA}_skillset

EXP_NAME="${DATA}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}"
OUTPUT_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
  LATEST_CKPT=$(ls -v "${OUTPUT_DIR}"/spline_vqae_epoch*.pt 2>/dev/null | tail -1 || true)
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
    --vae_type spline_vq \
    --latent_dim ${LATENT_DIM} \
    --num_embeddings ${NUM_EMBEDDINGS} \
    --commitment_cost 0.25 \
    --hidden_dim 128 \
    --num_layers 2 \
    --epochs 10000 \
    --lr 1.5e-3 \
    --batch_size 512 \
    --checkpoint_every 1000 \
    --wandb_project VAE_train \
    --wandb_run_name ${EXP_NAME} \
    ${RESUME_ARG}

echo "DONE ${EXP_NAME}"
