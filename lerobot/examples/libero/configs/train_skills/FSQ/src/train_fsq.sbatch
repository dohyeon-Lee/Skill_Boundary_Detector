#!/usr/bin/env bash
# Inputs:
#   FSQ inputs   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#   wrist tokens : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_wrist_pg{dino_patch_grid}.npz
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/outputs/FSQ/{fsq_run_name}
#   FSQ model    : {project_root}/outputs/FSQ/{fsq_run_name}/FSQ.pt
#   checkpoints  : {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt
#   skill tokens : {project_root}/outputs/FSQ/{fsq_run_name}/skill_latents.npz
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
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/fsq_config.yaml}

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
# Camera mode from fsq_terminator_use_third / use_wrist:
#   third=T wrist=F → 3rd-only   (primary=3rd)
#   third=T wrist=T → both       (primary=3rd, second=wrist)
#   third=F wrist=T → wrist-only (primary=wrist)
# LAZY: pass the per-frame DINO dir (NOT a materialized token npz). train_FSQ slices each skill's
# window on the fly (mmap) — no 27GB/camera token file, no extract step. The primary camera is picked
# by --image_key; in "both" mode the wrist is read from the SAME dir via --image_key_wrist.
if [ "${FSQ_TERMINATOR_USE_THIRD}" = "true" ]; then
  PRIMARY_IMAGE_KEY="observation.images.image"          # 3rd-person primary
elif [ "${FSQ_TERMINATOR_USE_WRIST}" = "true" ]; then
  PRIMARY_IMAGE_KEY="observation.images.wrist_image"     # wrist-only: wrist is the single (primary) camera
else
  echo "Both fsq_terminator_use_third and fsq_terminator_use_wrist are false — need ≥1 camera." >&2
  exit 1
fi
if [ ! -d "${FSQ_DINO_FEATURE_DIR}" ]; then
  echo "Per-frame DINO dir not found: ${FSQ_DINO_FEATURE_DIR}" >&2
  echo "Run build_data/submit_build_data.sh first (it prepares the per-frame DINO that lazy reads)." >&2
  exit 1
fi
DINO_ARGS=(--dino_features "${FSQ_DINO_FEATURE_DIR}" --image_key "${PRIMARY_IMAGE_KEY}")

# Terminator always uses the primary slot (--terminator_use_third = the dec_tokens camera); add the
# wrist slot only in "both" mode (wrist sliced from the same dir via --image_key_wrist).
TERM_ARGS=(--terminator_use_third)
if [ "${FSQ_TERMINATOR_USE_THIRD}" = "true" ] && [ "${FSQ_TERMINATOR_USE_WRIST}" = "true" ]; then
  TERM_ARGS+=(--terminator_use_wrist)
  DINO_ARGS+=(--image_key_wrist "observation.images.wrist_image")
else
  TERM_ARGS+=(--no-terminator_use_wrist)
fi

read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
N_TOKENS=$(( FSQ_PATCH_GRID * FSQ_PATCH_GRID + 1 ))

mkdir -p "${FSQ_OUTPUT_DIR}"

# Provenance so the eval can auto-evaluate this FSQ on the SAME skillset/cameras it was trained on,
# without the user re-specifying the DP (fsq_run_name doesn't encode the DP).
cat > "${FSQ_OUTPUT_DIR}/fsq_meta.json" <<JSON
{
  "dp_run_name": "${DP_POLICY}",
  "dp_checkpoint": "${DP_CHECKPOINT}",
  "fsq_patch_grid": ${FSQ_PATCH_GRID},
  "fsq_terminator_use_third": ${FSQ_TERMINATOR_USE_THIRD},
  "fsq_terminator_use_wrist": ${FSQ_TERMINATOR_USE_WRIST}
}
JSON

RESUME_ARG=()
LATEST_CKPT=$(ls -v "${FSQ_OUTPUT_DIR}"/FSQ_epoch*.pt 2>/dev/null | tail -1 || true)
if [ -n "${LATEST_CKPT}" ]; then
  RESUME_ARG=(--resume_from "${LATEST_CKPT}")
  echo "Resuming from ${LATEST_CKPT}"
else
  echo "No checkpoint found, training from scratch"
fi

# weighted_loss probe (tyro bool flag): end-weight the reconstructor delta loss.
WEIGHTED_ARG=(--no-weighted_loss)
if [ "${FSQ_WEIGHTED_LOSS:-false}" = "true" ]; then
  WEIGHTED_ARG=(--weighted_loss)
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/train_FSQ.py \
    --skills_dir              "${SKILLSET_DIR}/skills" \
    "${DINO_ARGS[@]}" \
    "${TERM_ARGS[@]}" \
    --output_dir              "${FSQ_OUTPUT_DIR}" \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --chunk_size              "${FSQ_CHUNK_SIZE}" \
    --hidden_dim              "${FSQ_HIDDEN_DIM}" \
    --num_layers              "${FSQ_NUM_LAYERS}" \
    --n_control               "${FSQ_N_CONTROL}" \
    --image_token_dim         "${FSQ_IMAGE_TOKEN_DIM}" \
    --feat_dim                "${DINO_FEATURE_DIM}" \
    --n_tokens                "${N_TOKENS}" \
    --image_model_name        "${DINO_IMAGE_MODEL_PATH}" \
    --image_size              "${DINO_IMAGE_SIZE}" \
    --patch_grid              "${FSQ_PATCH_GRID}" \
    --n_patch_raw             196 \
    --epochs                  "${FSQ_NUM_EPOCHS}" \
    --lr                      "${FSQ_LR}" \
    --batch_size              "${FSQ_BATCH_SIZE}" \
    --num_workers             "${FSQ_NUM_WORKERS}" \
    --delta_loss_weight       "${FSQ_DELTA_LOSS_WEIGHT}" \
    --progress_loss_weight    "${FSQ_PROGRESS_LOSS_WEIGHT}" \
    --end_loss_weight         "${FSQ_END_LOSS_WEIGHT}" \
    --end_target_sigma        "${FSQ_END_TARGET_SIGMA}" \
    "${WEIGHTED_ARG[@]}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE ${FSQ_RUN_NAME}"
