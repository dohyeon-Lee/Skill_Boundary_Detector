#!/usr/bin/env bash
#SBATCH --job-name=VQAE_enc_eval

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node01
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

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH=10000
NO_HTML=true

DATA=libero_90
LATENT_DIM=64
NUM_EMBEDDINGS=512
SKILLSET=${DATA}_skillset
# SKILLSET=${DATA}_skillset_fixed50

EXP="${DATA}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP}"

LATENTS_PATH="${SWEEP_DIR}/spline_vqae_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).npz"
OUTPUT_DIR="${SWEEP_DIR}/latent/vqae_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH})"
# ─────────────────────────────────────────────────────────────────────────────

echo "Experiment : ${EXP}"
echo "Checkpoint : epoch${CHECKPOINT_EPOCH}"
echo "Latents    : ${LATENTS_PATH}"

rm -rf ${OUTPUT_DIR}/wandb

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/visualize_latents.py \
    --latents_path ${LATENTS_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --color_by all \
    --n_clusters 15 \
    --dataset_dir ${HOMEDIR}${PROJDIR}/libero_dataset/libero_90 \
    --skills_per_cluster 5 \
    --wandb_project VAE_eval \
    --wandb_run_name ${EXP}_${SKILLSET} \
    $([ "${NO_HTML}" = "true" ] && echo "--no_html")

echo "DONE"
