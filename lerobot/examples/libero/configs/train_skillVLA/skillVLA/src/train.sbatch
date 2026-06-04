#!/usr/bin/env bash
# SkillVLA stage-3 PRE-TRAINING (PT).
# Trains policy.type=skill_vla on the build_data skillvla dataset, initialized from
# pi05_base, with the FROZEN FSQ end decoder. Resumes if a checkpoint exists.
#SBATCH --job-name=skillVLA_PT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR
# (the skillVLA dir the submit script cd'd into).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${SKILLVLA_TRAIN_CONFIG:-${SUBMIT_DIR}/skillVLA_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/skillVLA_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START SKILLVLA PT"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${SKILLVLA_DATASET_DIR}"
echo "FSQ      : ${FSQ_CKPT}"
echo "dino     : ${DINO_TOKENS_PATH}"
echo "output   : ${PT_OUTPUT_DIR}"
echo "lr/bs    : ${LR} / ${BATCH_SIZE}   detach(action/sp): ${DETACH_ACTION_PREFIX_GRAD}/${DETACH_SP_PREFIX}"
nvidia-smi || true

for p in "${SKILLVLA_DATASET_DIR}" "${FSQ_CKPT}" "${DINO_TOKENS_PATH}"; do
  if [ ! -e "${p}" ]; then
    echo "Missing required artifact: ${p}  (run configs/train_skillVLA/build_data first)" >&2
    exit 1
  fi
done

# Resume if a checkpoint exists; otherwise wipe a stale (incomplete) output dir.
CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found → resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "No checkpoint → removing stale ${PT_OUTPUT_DIR} and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla \
    --policy.pretrained_path="${PI_BASE}" \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.vae_decoder_path="${FSQ_CKPT}" \
    --policy.skill_decoder_image_model_name="${IMAGE_MODEL_PATH}" \
    --policy.skill_decoder_dino_tokens_path="${DINO_TOKENS_PATH}" \
    --policy.skill_decoder_dino_cache_path="${DINO_CACHE_PATH}" \
    --policy.skill_decoder_dino_build_cache=true \
    --policy.skill_predictor_loss_weight="${SKILL_PREDICTOR_LOSS_WEIGHT}" \
    --policy.skill_decoder_end_threshold="${SKILL_DECODER_END_THRESHOLD}" \
    --policy.skill_decoder_state_indices="${SKILL_DECODER_STATE_INDICES}" \
    --policy.use_reconstructor_chunk_suffix="${USE_RECONSTRUCTOR_CHUNK_SUFFIX}" \
    --policy.skill_boundary_random_p="${SKILL_BOUNDARY_RANDOM_P}" \
    --policy.detach_action_prefix_grad="${DETACH_ACTION_PREFIX_GRAD}" \
    --policy.detach_sp_prefix="${DETACH_SP_PREFIX}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.optimizer_lr="${LR}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE → ${PT_OUTPUT_DIR}"
