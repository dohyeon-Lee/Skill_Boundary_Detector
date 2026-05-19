#!/usr/bin/env bash
#SBATCH --job-name=FSQ_codebook_vis
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then EEF_MODE=zero; else EEF_MODE=abs; fi

FSQ_TAG="fsq555"
MODE_TAG="${DECODER_MODE}K${CHUNK_SIZE}"
EXP_NAME="${DATA}_${FSQ_TAG}_${IMAGE_EXP_TAG}_${EEF_MODE}_${MODE_TAG}"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_latents/${EXP_NAME}"

if [ "${CHECKPOINT_EPOCH}" -eq 0 ]; then
  LATENTS_PATH="${SWEEP_DIR}/skill_latents.npz"
  EPOCH_TAG="best"
else
  # checkpoint_every 저장 시 별도 latents 파일이 없으므로 기본 파일 사용
  LATENTS_PATH="${SWEEP_DIR}/skill_latents.npz"
  EPOCH_TAG="epoch$(printf '%04d' ${CHECKPOINT_EPOCH})"
fi

OUTPUT_DIR="${SWEEP_DIR}/codebook_vis/${EPOCH_TAG}"
# ─────────────────────────────────────────────────────────────────────────────

echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_DIR}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/codebook_visualizer.py \
    --latents_path      ${LATENTS_PATH} \
    --dataset_dir       ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA} \
    --num_embeddings    125 \
    --thumb_size        96 \
    --max_per_entry     50 \
    --wandb_project     VAE_eval \
    --wandb_run_name    ${EXP_NAME}_codebook_${EPOCH_TAG}

echo "DONE → ${OUTPUT_DIR}"
