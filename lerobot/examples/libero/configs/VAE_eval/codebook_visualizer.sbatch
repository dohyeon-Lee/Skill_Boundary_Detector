#!/usr/bin/env bash
#SBATCH --job-name=VQAE_codebook_vis
#SBATCH --partition=debug
#SBATCH --nodelist=node01 #,node01
#SBATCH --gres=gpu:1
#SBATCH --qos=big_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate
unset LD_LIBRARY_PATH

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH=2000
DATA=libero_90
LATENT_DIM=128
NUM_EMBEDDINGS=128
ENCODER_TYPE=gru
DECODER_RNN_TYPE=mlp
ZERO_START_EEF=true
END_POS_WEIGHT=1
COMMITMENT_COST=0.1   # default 0.25;
VISUAL_BACKBONE=dinov3_vits16 # dinov2_small
SKILLSET=${DATA}_skillset

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then EEF_MODE=zero; else EEF_MODE=abs; fi
EXP="${DATA}_vqvae_${IMAGE_EXP_TAG}_${ENCODER_TYPE}_${DECODER_RNN_TYPE}_${EEF_MODE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_${COMMITMENT_COST}_exp3"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP}"
LATENTS_PATH="${SWEEP_DIR}/spline_vqae_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).npz"
OUTPUT_DIR="${SWEEP_DIR}/codebook_vis/epoch$(printf '%04d' ${CHECKPOINT_EPOCH})"
# ─────────────────────────────────────────────────────────────────────────────

echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_DIR}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/codebook_visualizer.py \
    --latents_path      ${LATENTS_PATH} \
    --dataset_dir       ${HOMEDIR}${PROJDIR}/libero_dataset/${DATA} \
    --num_embeddings    ${NUM_EMBEDDINGS} \
    --thumb_size        96 \
    --max_per_entry     50 \
    --wandb_project     VAE_eval \
    --wandb_run_name    ${EXP}_codebook_epoch${CHECKPOINT_EPOCH}

echo "DONE → ${OUTPUT_DIR}"
