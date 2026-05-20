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
CHECKPOINT_EPOCH=500      # 0 → best checkpoint (FSQ.pt), else FSQ_epoch{N:04d}.pt
DATA=libero_90
DATADIR=libero_dataset
VISUAL_BACKBONE=dinov3_vits16
SKILLSET=${DATA}_skillset
DERIVED_DATA_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data
FSQ_PRECOMPUTE_DIR=${DERIVED_DATA_DIR}/${DATA}_for_FSQ
SKILLSET_DIR=${FSQ_PRECOMPUTE_DIR}/${SKILLSET}

DECODER_MODE=single_step
DECODER_IMAGE_MODE=dino_flags
CHUNK_SIZE=10
FSQ_LEVELS=(5 5 5)
ZERO_START_EEF=false
END_THRESHOLD=0.5
MAX_PLOT_SAMPLES=5
MAX_PLOT_ENTRIES=20

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then
  EEF_MODE=zero
  ZERO_START_ARG="--zero_start_eef"
else
  EEF_MODE=abs
  ZERO_START_ARG="--no-zero_start_eef"
fi

FSQ_TAG="fsq$(IFS=; echo "${FSQ_LEVELS[*]}")"
FSQ_CODEBOOK_SIZE=1
for level in "${FSQ_LEVELS[@]}"; do
  FSQ_CODEBOOK_SIZE=$((FSQ_CODEBOOK_SIZE * level))
done
if [ "${DECODER_MODE}" = "chunk" ]; then
  MODE_TAG="chunkK${CHUNK_SIZE}"
else
  MODE_TAG="single_step"
fi
EXP_NAME="${DATA}_${FSQ_TAG}_${IMAGE_EXP_TAG}_${EEF_MODE}_${MODE_TAG}_${DECODER_IMAGE_MODE}"
SWEEP_DIR="${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_FSQ/${EXP_NAME}"

if [ "${CHECKPOINT_EPOCH}" -eq 0 ]; then
  MODEL_PATH="${SWEEP_DIR}/FSQ.pt"
  LATENTS_PATH="${SWEEP_DIR}/skill_latents.npz"
  EPOCH_TAG="best"
else
  MODEL_PATH="${SWEEP_DIR}/FSQ_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).pt"
  LATENTS_PATH="${SWEEP_DIR}/skill_latents_epoch$(printf '%04d' ${CHECKPOINT_EPOCH}).npz"
  EPOCH_TAG="epoch$(printf '%04d' ${CHECKPOINT_EPOCH})"
fi

SKILLS_DIR="${SKILLSET_DIR}/skills"
DINO_TOKENS_PATH="${FSQ_PRECOMPUTE_DIR}/${IMAGE_FEATURE_TAG}_tokens.npz"
SAM2_FLAGS_PATH="${FSQ_PRECOMPUTE_DIR}/patch_flags.npz"
SAM2_MASKS_DIR="${FSQ_PRECOMPUTE_DIR}/sam2_masks"
if [ -f "${SAM2_FLAGS_PATH}" ]; then
  SAM2_INPUT="${SAM2_FLAGS_PATH}"
else
  SAM2_INPUT="${SAM2_MASKS_DIR}"
fi
OUTPUT_HTML="${SWEEP_DIR}/decoder_eval/${EPOCH_TAG}.html"
# ─────────────────────────────────────────────────────────────────────────────

echo "Model   : ${MODEL_PATH}"
echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_HTML}"

if [ ! -f "${LATENTS_PATH}" ]; then
  echo "Latents not found; encoding skills from checkpoint..."
  PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
    python examples/libero/encode_FSQ_skills.py \
      --model_path      ${MODEL_PATH} \
      --skills_dir      ${SKILLS_DIR} \
      --dino_features   ${DINO_TOKENS_PATH} \
      --output_path     ${LATENTS_PATH} \
      ${ZERO_START_ARG} \
      --device          cuda
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/decoder_eval.py \
    --model_path            ${MODEL_PATH} \
    --latents_path          ${LATENTS_PATH} \
    --skills_dir            ${SKILLS_DIR} \
    --dino_features         ${DINO_TOKENS_PATH} \
    --sam2_masks_dir        ${SAM2_INPUT} \
    ${ZERO_START_ARG} \
    --end_threshold         ${END_THRESHOLD} \
    --max_plot_samples      ${MAX_PLOT_SAMPLES} \
    --max_plot_entries      ${MAX_PLOT_ENTRIES} \
    --wandb_project         VAE_eval \
    --wandb_run_name        ${EXP_NAME}_decoder_${EPOCH_TAG} \
    --output_html           ${OUTPUT_HTML}

echo "DONE → ${OUTPUT_HTML}"
