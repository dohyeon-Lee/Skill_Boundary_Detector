#!/usr/bin/env bash
#SBATCH --job-name=FSQ_codebook_vis

#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --partition=asus_pro6000
#SBATCH --qos=pro6000_qos

# SBATCH --partition=debug
# SBATCH --nodelist=node100
# SBATCH --qos=big_qos

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
[ ! -f "${CONFIG_PY}" ] && CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
eval "$(python3 "${CONFIG_PY}" --shell | grep -E '^export (HOMEDIR|PROJDIR)=')"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate
unset LD_LIBRARY_PATH

# ── 설정 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_EPOCH=1000      # 0 → best checkpoint (FSQ.pt), else FSQ_epoch{N:04d}.pt
DATA=libero_90
DATADIR=libero_dataset
VISUAL_BACKBONE=dinov3_vits16
SKILLSET=${DATA}_skillset_prev
DERIVED_DATA_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data
FSQ_PRECOMPUTE_DIR=${DERIVED_DATA_DIR}/${DATA}_for_FSQ_prev
SKILLSET_DIR=${FSQ_PRECOMPUTE_DIR}/${SKILLSET}

DECODER_MODE=chunk
DECODER_IMAGE_MODE=dino_only
CHUNK_SIZE=10
FSQ_LEVELS=(5 5 5)
ZERO_START_EEF=false

source examples/libero/configs/VAE_train/visual_backbone.sh
resolve_visual_backbone

if [ "${ZERO_START_EEF}" = "true" ]; then EEF_MODE=zero; else EEF_MODE=abs; fi

FSQ_TAG="fsq$(IFS=; echo "${FSQ_LEVELS[*]}")"
FSQ_CODEBOOK_SIZE=1
for level in "${FSQ_LEVELS[@]}"; do
  FSQ_CODEBOOK_SIZE=$((FSQ_CODEBOOK_SIZE * level))
done
if [ "${DECODER_MODE}" = "chunk" ]; then
  MODE_TAG="chunk"
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

OUTPUT_DIR="${SWEEP_DIR}/codebook_vis/${EPOCH_TAG}"
DINO_TOKENS_PATH="${FSQ_PRECOMPUTE_DIR}/${IMAGE_FEATURE_TAG}_tokens.npz"
# ─────────────────────────────────────────────────────────────────────────────

echo "Model   : ${MODEL_PATH}"
echo "Latents : ${LATENTS_PATH}"
echo "Skills  : ${SKILLSET_DIR}/skills"
echo "DINO    : ${DINO_TOKENS_PATH}"
echo "Output  : ${OUTPUT_DIR}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: model checkpoint not found: ${MODEL_PATH}" >&2
  echo "Check EXP_NAME=${EXP_NAME}" >&2
  exit 1
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "ERROR: skills directory not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "ERROR: DINO token file not found: ${DINO_TOKENS_PATH}" >&2
  exit 1
fi

if [ ! -f "${LATENTS_PATH}" ]; then
  echo "Latents not found; encoding skills from checkpoint..."
  if [ "${ZERO_START_EEF}" = "true" ]; then
    ZERO_START_ARG="--zero_start_eef"
  else
    ZERO_START_ARG="--no-zero_start_eef"
  fi
  PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
    python examples/libero/encode_FSQ_skills.py \
      --model_path      ${MODEL_PATH} \
      --skills_dir      ${SKILLSET_DIR}/skills \
      --dino_features   ${DINO_TOKENS_PATH} \
      --output_path     ${LATENTS_PATH} \
      ${ZERO_START_ARG} \
      --device          cuda
fi

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/codebook_visualizer.py \
    --latents_path      ${LATENTS_PATH} \
    --dataset_dir       ${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA} \
    --num_embeddings    ${FSQ_CODEBOOK_SIZE} \
    --fsq_levels        "${FSQ_LEVELS[@]}" \
    --thumb_size        96 \
    --max_per_entry     20 \
    --output_html       ${OUTPUT_DIR}/codebook_vis.html \
    --wandb_project     VAE_eval \
    --wandb_run_name    ${EXP_NAME}_codebook_${EPOCH_TAG}

echo "DONE → ${OUTPUT_DIR}"
