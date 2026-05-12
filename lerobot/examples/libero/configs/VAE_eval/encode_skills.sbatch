#!/usr/bin/env bash
#SBATCH --job-name=VQAE_encode
#SBATCH --partition=debug
#SBATCH --nodelist=node01
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
# VQ-VAE 학습에 사용한 모델 설정 (체크포인트와 반드시 일치해야 함)
CHECKPOINT_EPOCH=1500
LATENT_DIM=128
NUM_EMBEDDINGS=128
ENCODER_TYPE=gru
DECODER_RNN_TYPE=mlp
ZERO_START_EEF=true
END_POS_WEIGHT=1
COMMITMENT_COST=0.1
VISUAL_BACKBONE=dinov3_vits16
TRAIN_DATA=libero_90          # VQ-VAE를 학습한 데이터

# 인코딩할 새 데이터 (학습 때 쓰지 않은 데이터도 가능)
# libero_10, libero_10_op1_50, libero_10_op1_10
ENCODE_DATA=libero_10
SKILLDIR=libero_dataset # libero_dataset, libero_small_dataset
SKILLSET=${ENCODE_DATA}_skillset

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then EEF_MODE=zero; else EEF_MODE=abs; fi
EXP="${TRAIN_DATA}_vqvae_${IMAGE_EXP_TAG}_${ENCODER_TYPE}_${DECODER_RNN_TYPE}_${EEF_MODE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_${COMMITMENT_COST}_exp3"

MODEL_PATH="${HOMEDIR}${PROJDIR}/outputs/${TRAIN_DATA}_skillset_latents/${EXP}/spline_vqae_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).pt"
SKILLS_DIR="${HOMEDIR}${PROJDIR}/${SKILLDIR}/${SKILLSET}"
OUTPUT_PATH="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP}/spline_vqae_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).npz"
# ─────────────────────────────────────────────────────────────────────────────

echo "Model   : ${MODEL_PATH}"
echo "Skills  : ${SKILLS_DIR}"
echo "Output  : ${OUTPUT_PATH}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/encode_skills.py \
    --model_path       ${MODEL_PATH} \
    --skills_dir       ${SKILLS_DIR} \
    --output_path      ${OUTPUT_PATH} \
    --zero_start_eef

echo "DONE → ${OUTPUT_PATH}"
