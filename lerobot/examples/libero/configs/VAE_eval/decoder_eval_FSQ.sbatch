#!/usr/bin/env bash
#SBATCH --job-name=FSQ_decoder_eval
#SBATCH --partition=debug
#SBATCH --nodelist=node100
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
CHECKPOINT_EPOCH=2000      # 0 → best checkpoint (FSQ.pt), else FSQ_epoch{N:04d}.pt
DATA=libero_90
DATADIR=libero_dataset
VISUAL_BACKBONE=dinov3_vits16
SKILLSET=${DATA}_skillset

DECODER_MODE=chunk
CHUNK_SIZE=10
ZERO_START_EEF=true
END_THRESHOLD=0.5
MAX_PLOT_SAMPLES=20

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then
  EEF_MODE=zero
  ZERO_START_ARG="--zero_start_eef"
else
  EEF_MODE=abs
  ZERO_START_ARG="--no-zero_start_eef"
fi

FSQ_TAG="fsq555"
MODE_TAG="${DECODER_MODE}K${CHUNK_SIZE}"
EXP_NAME="${DATA}_${FSQ_TAG}_${IMAGE_EXP_TAG}_${EEF_MODE}_${MODE_TAG}"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

if [ "${CHECKPOINT_EPOCH}" -eq 0 ]; then
  MODEL_PATH="${SWEEP_DIR}/FSQ.pt"
  EPOCH_TAG="best"
else
  MODEL_PATH="${SWEEP_DIR}/FSQ_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).pt"
  EPOCH_TAG="epoch$(printf '%04d' ${CHECKPOINT_EPOCH})"
fi

LATENTS_PATH="${SWEEP_DIR}/skill_latents.npz"
SKILLS_DIR="${HOMEDIR}${PROJDIR}/${DATADIR}/${SKILLSET}/skills"
DINO_TOKENS_PATH="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_${VISUAL_BACKBONE}_tokens.npz"
SAM2_MASKS_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_FSQ"
OUTPUT_HTML="${SWEEP_DIR}/decoder_eval/${EPOCH_TAG}.html"
# ─────────────────────────────────────────────────────────────────────────────

echo "Model   : ${MODEL_PATH}"
echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_HTML}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/decoder_eval.py \
    --model_path            ${MODEL_PATH} \
    --latents_path          ${LATENTS_PATH} \
    --skills_dir            ${SKILLS_DIR} \
    --dino_features         ${DINO_TOKENS_PATH} \
    --sam2_masks_dir        ${SAM2_MASKS_DIR} \
    ${ZERO_START_ARG} \
    --end_threshold         ${END_THRESHOLD} \
    --max_plot_samples      ${MAX_PLOT_SAMPLES} \
    --wandb_project         VAE_eval \
    --wandb_run_name        ${EXP_NAME}_decoder_${EPOCH_TAG} \
    --output_html           ${OUTPUT_HTML}

echo "DONE → ${OUTPUT_HTML}"
