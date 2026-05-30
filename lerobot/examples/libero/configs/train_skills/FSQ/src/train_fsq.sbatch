#!/usr/bin/env bash
# Inputs:
#   FSQ inputs   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#   SAM2 flags   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/patch_flags.npz
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/FSQ_outputs/{fsq_run_name}
#   FSQ model    : {project_root}/FSQ_outputs/{fsq_run_name}/FSQ.pt
#   checkpoints  : {project_root}/FSQ_outputs/{fsq_run_name}/FSQ_epoch*.pt
#   skill tokens : {project_root}/FSQ_outputs/{fsq_run_name}/skill_latents.npz
#
# Train FSQ from prepared FSQ_inputs.
#SBATCH --job-name=FSQ_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs_train/%x_%j.out
#SBATCH --error=logs_train/%x_%j.err
set -euo pipefail
mkdir -p logs_train

# Under Slurm the script runs from a spool copy, so ${BASH_SOURCE[0]} is unreliable.
# Anchor on SLURM_SUBMIT_DIR (the FSQ dir the submit script cd'd into); fall back to
# this file's parent for direct execution.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMMON_SRC_DIR="${SUBMIT_DIR}/../src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/../train_skills_config.yaml}

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=""
# Reduce CUDA fragmentation: activation tensor sizes vary per batch (max skill length).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${PROJECT_ROOT}/../.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${PROJECT_ROOT}/../.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START FSQ TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Inputs     : ${FSQ_INPUTS_DIR}"
echo "Output     : ${FSQ_OUTPUT_DIR}"
nvidia-smi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi
if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "DINO tokens not found: ${DINO_TOKENS_PATH}" >&2
  exit 1
fi

SAM2_ARGS=()
if [ "${DECODER_IMAGE_MODE}" = "dino_flags" ]; then
  if [ -f "${SAM2_FLAGS_PATH}" ]; then
    SAM2_ARGS=(--sam2_masks_dir "${SAM2_FLAGS_PATH}")
  elif [ -d "${SAM2_MASKS_DIR}" ]; then
    SAM2_ARGS=(--sam2_masks_dir "${SAM2_MASKS_DIR}")
  else
    echo "Patch flags not found for decoder_image_mode=dino_flags: ${SAM2_FLAGS_PATH}" >&2
    exit 1
  fi
fi

read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
N_TOKENS=$(( DINO_PATCH_GRID * DINO_PATCH_GRID + 1 ))

mkdir -p "${FSQ_OUTPUT_DIR}"

RESUME_ARG=()
LATEST_CKPT=$(ls -v "${FSQ_OUTPUT_DIR}"/FSQ_epoch*.pt 2>/dev/null | tail -1 || true)
if [ -n "${LATEST_CKPT}" ]; then
  RESUME_ARG=(--resume_from "${LATEST_CKPT}")
  echo "Resuming from ${LATEST_CKPT}"
else
  echo "No checkpoint found, training from scratch"
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/train_FSQ.py \
    --skills_dir              "${SKILLSET_DIR}/skills" \
    --dino_features           "${DINO_TOKENS_PATH}" \
    "${SAM2_ARGS[@]}" \
    --output_dir              "${FSQ_OUTPUT_DIR}" \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --decoder_image_mode      "${DECODER_IMAGE_MODE}" \
    --chunk_size              "${FSQ_CHUNK_SIZE}" \
    --hidden_dim              "${FSQ_HIDDEN_DIM}" \
    --num_layers              "${FSQ_NUM_LAYERS}" \
    --n_control               "${FSQ_N_CONTROL}" \
    --image_token_dim         "${FSQ_IMAGE_TOKEN_DIM}" \
    --feat_dim                "${DINO_FEATURE_DIM}" \
    --n_tokens                "${N_TOKENS}" \
    --image_model_name        "${DINO_IMAGE_MODEL_PATH}" \
    --image_size              "${DINO_IMAGE_SIZE}" \
    --patch_grid              "${DINO_PATCH_GRID}" \
    --n_patch_raw             196 \
    --epochs                  "${FSQ_NUM_EPOCHS}" \
    --lr                      "${FSQ_LR}" \
    --batch_size              "${FSQ_BATCH_SIZE}" \
    --delta_loss_weight       "${FSQ_DELTA_LOSS_WEIGHT}" \
    --progress_loss_weight    "${FSQ_PROGRESS_LOSS_WEIGHT}" \
    --end_loss_weight         "${FSQ_END_LOSS_WEIGHT}" \
    --max_length              "${FSQ_MAX_LENGTH}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE ${FSQ_RUN_NAME}"
