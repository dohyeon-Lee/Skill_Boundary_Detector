#!/usr/bin/env bash
#SBATCH --job-name=VQAE_decoder_eval
#SBATCH --partition=debug
#SBATCH --nodelist=node100 #,node01
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
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
END_THRESHOLD=0.5
END_POS_WEIGHT=1
COMMITMENT_COST=0.1   # default 0.25;
MAX_PLOT_SAMPLES=20
VISUAL_BACKBONE=dinov3_vits16
SKILLSET=${DATA}_skillset

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then
  EEF_MODE=zero
  ZERO_START_ARG="--zero_start_eef"
else
  EEF_MODE=abs
  ZERO_START_ARG="--no-zero_start_eef"
fi
EXP="${DATA}_vqvae_${IMAGE_EXP_TAG}_${ENCODER_TYPE}_${DECODER_RNN_TYPE}_${EEF_MODE}_lat${LATENT_DIM}_vq${NUM_EMBEDDINGS}_${COMMITMENT_COST}_exp3"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP}"
MODEL_PATH="${SWEEP_DIR}/spline_vqae_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).pt"
LATENTS_PATH="${SWEEP_DIR}/spline_vqae_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).npz"
SKILLS_DIR="${HOMEDIR}${PROJDIR}/libero_dataset/${SKILLSET}"
OUTPUT_HTML="${SWEEP_DIR}/decoder_eval/epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).html"
# ─────────────────────────────────────────────────────────────────────────────

echo "Model   : ${MODEL_PATH}"
echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_HTML}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/decoder_eval.py \
    --model_path            ${MODEL_PATH} \
    --latents_path          ${LATENTS_PATH} \
    --skills_dir            ${SKILLS_DIR} \
    --image_features_path   ${IMAGE_FEATURES_PATH} \
    ${ZERO_START_ARG} \
    --end_threshold         ${END_THRESHOLD} \
    --max_plot_samples      ${MAX_PLOT_SAMPLES} \
    --wandb_project         VAE_eval \
    --wandb_run_name        ${EXP}_decoder_epoch${CHECKPOINT_EPOCH} \
    --output_html           ${OUTPUT_HTML}

echo "DONE → ${OUTPUT_HTML}"
