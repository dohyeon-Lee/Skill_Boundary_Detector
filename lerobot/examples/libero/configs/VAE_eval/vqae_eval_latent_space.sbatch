#!/usr/bin/env bash
#SBATCH --job-name=VQAE_latent_eval

#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate

echo "START JOB"
echo "Node: $SLURMD_NODENAME"

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
LATENT_DIM=64
NUM_EMBEDDINGS=512
STEP=5000
FIXED=50

# A: VQAE fixed-length skillset
FIXED_DIR=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_fixed${FIXED}_latents/libero_90_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}
FIXED_NPZ=${FIXED_DIR}/spline_vqae_latents_epoch$(printf '%04d' ${STEP}).npz
FIXED_PT=${FIXED_DIR}/spline_vqae_epoch$(printf '%04d' ${STEP}).pt

# B: VQAE variable-length skillset
VARIABLE_DIR=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}
VARIABLE_NPZ=${VARIABLE_DIR}/spline_vqae_latents_epoch$(printf '%04d' ${STEP}).npz
VARIABLE_PT=${VARIABLE_DIR}/spline_vqae_epoch$(printf '%04d' ${STEP}).pt

DATASET=${HOMEDIR}${PROJDIR}/libero_dataset/libero_90_skillvla

OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/latent_eval_vqae

WANDB_PROJECT=VAE_eval
WANDB_JOB=vqae_latent_eval_libero90

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

# ── 평가 실행 ─────────────────────────────────────────────────────────────────
for K in 20; do
    echo ""
    echo "=== K=${K} clusters ==="
    python examples/libero/eval_latent_space.py \
        --npz_A   ${FIXED_NPZ} \
        --vae_A   ${FIXED_PT} \
        --name_A  fixed${FIXED} \
        --npz_B   ${VARIABLE_NPZ} \
        --vae_B   ${VARIABLE_PT} \
        --name_B  variable \
        --dataset_path ${DATASET} \
        --output_dir   ${OUTPUT_DIR} \
        --n_clusters   ${K} \
        --n_sample     20 \
        --n_resample   80 \
        --n_recon_eval 500 \
        --seed         42 \
        --wandb_enable \
        --wandb_project  ${WANDB_PROJECT} \
        --wandb_job_name ${WANDB_JOB}_k${K}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_fixed${FIXED}_ep${STEP}
done

echo "DONE"
