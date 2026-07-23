#!/usr/bin/env bash
# Inputs:
#   FSQ inputs   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
#   DINO tokens  : (ONLINE — raw dataset의 mp4에서 학습 시작 시 warm-pass 인코딩; 디스크 산출물 없음)
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
# Camera mode from fsq_terminator_use_third / use_wrist. The saved cfg flags are FAITHFUL to these, and
# the camera→slot routing matches how EVERY downstream consumer feeds the terminator: 3rd-person →
# primary slot (dec_tokens, terminator_use_third), wrist → wrist slot (dec_tokens_wrist, use_wrist).
#   third=T wrist=F → 3rd-only  : primary=3rd (used)
#   third=T wrist=T → both      : primary=3rd (used) + wrist slot=wrist
#   third=F wrist=T → wrist-only: wrist slot=wrist (used); primary is loaded but the terminator IGNORES
#                                 it (third=F) — so downstream, which always puts 3rd in the primary slot
#                                 and gates wrist by use_wrist, feeds the CORRECT camera. (Historic bug:
#                                 wrist-only used to be saved as third=T with wrist in the primary slot,
#                                 so downstream fed it 3rd where it was trained on wrist.)
# ONLINE: DINO 토큰은 train_FSQ가 학습 시작 시 raw mp4에서 직접 warm-pass 인코딩
# (디스크 precompute 완전 제거 — per-frame DINO dir/토큰 npz 불필요, dino-prep 빌드 단계 없음).
# primary via --image_key; wrist slot via --image_key_wrist (두 키가 같으면 train_FSQ가 클립 공유).
if [ "${FSQ_TERMINATOR_USE_THIRD}" != "true" ] && [ "${FSQ_TERMINATOR_USE_WRIST}" != "true" ]; then
  echo "Both fsq_terminator_use_third and fsq_terminator_use_wrist are false — need ≥1 camera." >&2
  exit 1
fi
if [ ! -d "${RAW_DATASET_DIR}/videos" ]; then
  echo "Raw dataset videos not found: ${RAW_DATASET_DIR}/videos (online DINO warm-pass의 원천)" >&2
  exit 1
fi
# Primary slot camera: 3rd when the terminator uses it; for wrist-only set it to wrist (loaded but
# unused by the terminator) so a wrist-only run does not require a 3rd-person camera to exist.
if [ "${FSQ_TERMINATOR_USE_THIRD}" = "true" ]; then
  PRIMARY_IMAGE_KEY="observation.images.image"
else
  PRIMARY_IMAGE_KEY="observation.images.wrist_image"
fi
DINO_ARGS=(--raw_dataset_dir "${RAW_DATASET_DIR}" --image_key "${PRIMARY_IMAGE_KEY}")

# Faithful flags: each --terminator_use_* reflects the REAL config (no more hardcoded --terminator_use_third).
if [ "${FSQ_TERMINATOR_USE_THIRD}" = "true" ]; then
  TERM_ARGS=(--terminator_use_third)
else
  TERM_ARGS=(--no-terminator_use_third)
fi
if [ "${FSQ_TERMINATOR_USE_WRIST}" = "true" ]; then
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
  "fsq_skill_cond_mode": "${FSQ_SKILL_COND_MODE}",
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

# best-val SELECTION override: pass each val_select weight only if set in config (empty → omit → the
# trainer falls back to selecting FSQ.pt on the actual loss = delta+progress+end with their weights).
VAL_SELECT_ARGS=()
if [ -n "${FSQ_VAL_SELECT_DELTA_WEIGHT:-}" ];    then VAL_SELECT_ARGS+=(--val_select_delta_weight    "${FSQ_VAL_SELECT_DELTA_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_PROGRESS_WEIGHT:-}" ]; then VAL_SELECT_ARGS+=(--val_select_progress_weight "${FSQ_VAL_SELECT_PROGRESS_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_END_WEIGHT:-}" ];      then VAL_SELECT_ARGS+=(--val_select_end_weight      "${FSQ_VAL_SELECT_END_WEIGHT}"); fi

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
    --skill_cond_mode         "${FSQ_SKILL_COND_MODE}" \
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
    "${VAL_SELECT_ARGS[@]}" \
    --end_target_sigma        "${FSQ_END_TARGET_SIGMA}" \
    "${WEIGHTED_ARG[@]}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE ${FSQ_RUN_NAME}"
