#!/usr/bin/env bash
#SBATCH --job-name=latent_eval

#SBATCH --partition=debug
#SBATCH --nodelist=node100
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
STEP=10000

FIXED50_NPZ=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_fixed50_latents/libero_90_lat${LATENT_DIM}_spline/spline_vae_latents_epoch${STEP}.npz
FIXED50_VAE=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_fixed50_latents/libero_90_lat${LATENT_DIM}_spline/spline_vae_epoch${STEP}.pt

VARIABLE_NPZ=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat${LATENT_DIM}_spline/spline_vae_latents_epoch${STEP}.npz
VARIABLE_VAE=${HOMEDIR}${PROJDIR}/outputs/libero_90_skillset_latents/libero_90_lat${LATENT_DIM}_spline/spline_vae_epoch${STEP}.pt

DATASET=${HOMEDIR}${PROJDIR}/libero_dataset/libero_90_skillvla

OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/latent_eval

WANDB_PROJECT=VAE_eval
WANDB_JOB=latent_eval_libero90

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

# ── 평가 실행 ─────────────────────────────────────────────────────────────────
for K in 20 30 40; do
    echo ""
    echo "=== K=${K} clusters ==="
    python examples/libero/eval_latent_space.py \
        --npz_A   ${FIXED50_NPZ} \
        --vae_A   ${FIXED50_VAE} \
        --name_A  fixed50 \
        --npz_B   ${VARIABLE_NPZ} \
        --vae_B   ${VARIABLE_VAE} \
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
        --wandb_job_name ${WANDB_JOB}_k${K}_${LATENT_DIM}_${STEP}
done

echo "DONE"
