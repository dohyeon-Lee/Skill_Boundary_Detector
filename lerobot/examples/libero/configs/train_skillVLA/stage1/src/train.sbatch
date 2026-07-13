#!/usr/bin/env bash
# SkillVLA Stage-1 training: policy.type=skill_expert (standalone action expert).
# Trains the action expert by flow matching on the build_data skillvla dataset (raw images +
# skill code + actions). Warm-starts from pi05_base (or scratch). Resumes if a checkpoint exists.
#SBATCH --job-name=stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR (the stage1 dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SUBMIT_DIR}/stage1_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START STAGE-1 (skill_expert)"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${SKILLVLA_DATASET_DIR}"
echo "init     : ${PI_BASE:-scratch}   dino: ${DINO_MODEL_PATH}  dino_lr: ${DINO_LR:-(same)}"
echo "output   : ${PT_OUTPUT_DIR}   lr/bs: ${LR}/${BATCH_SIZE}"
nvidia-smi || true

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}  (run configs/train_skillVLA/build_data first)" >&2
  exit 1
fi

# Optional args: pi05 warm-start and a separate vision LR (empty → omitted).
EXTRA_ARGS=()
if [ -n "${PI_BASE}" ]; then
  EXTRA_ARGS+=(--policy.pretrained_path="${PI_BASE}")
fi
if [ -n "${DINO_LR}" ]; then
  EXTRA_ARGS+=(--policy.dino_lr="${DINO_LR}")
fi
if [ -n "${SIGLIP_LR}" ]; then
  EXTRA_ARGS+=(--policy.siglip_lr="${SIGLIP_LR}")
fi
if [ -n "${COND_ENCODER_VARIANT}" ]; then
  EXTRA_ARGS+=(--policy.cond_encoder_variant="${COND_ENCODER_VARIANT}")
fi
# Co-trained terminator: ONLINE DINO (배치의 현재 프레임 이미지를 라이브 토큰화 — dino.npz 인자 은퇴).
# DINO 모델 경로만 로컬로 오버라이드 (FSQ ckpt에 박힌 원 학습 머신 절대경로 방어).
if [ "${TRAIN_TERMINATOR:-false}" = "true" ] && [ -n "${TERMINATOR_DINO_MODEL_PATH:-}" ]; then
  EXTRA_ARGS+=(--policy.terminator_dino_model_path="${TERMINATOR_DINO_MODEL_PATH}")
fi

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
    --policy.type=skill_expert \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.state_cond_mode="${STATE_COND_MODE}" \
    --policy.vision_backbone="${VISION_BACKBONE}" \
    --policy.dino_model_path="${DINO_MODEL_PATH}" \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.siglip_image_size="${SIGLIP_IMAGE_SIZE}" \
    --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.skill_end_loss_weight="${SKILL_END_LOSS_WEIGHT}" \
    --policy.action_weight="${ACTION_WEIGHT}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.fsq_path="${FSQ_PATH}" \
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}" \
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}" \
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.optimizer_lr="${LR}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE → ${PT_OUTPUT_DIR}"
