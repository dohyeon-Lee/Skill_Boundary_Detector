#!/usr/bin/env bash
# SkillVLA Stage-2 training: policy.type=skill_vla.
# VLM (PaliGemma, warm-started from pi05_base) predicts the skill; an action expert warm-started
# from a Stage-1 skill_expert checkpoint flow-matches the action chunk. expert_arch (A/B) /
# vision_backbone / skill_vocab are read from the Stage-1 checkpoint config by the model. Resumes
# from a Stage-2 checkpoint if one exists.
#SBATCH --job-name=stage2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR (the stage2 dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE2_TRAIN_CONFIG:-${SUBMIT_DIR}/stage2_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START STAGE-2 (skill_vla)"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${SKILLVLA_DATASET_DIR}"
echo "VLM init : ${PI_BASE}"
echo "expert   : ${STAGE1_CHECKPOINT_PATH}  (arch=joint)"
echo "FSQ(eval): ${FSQ_CKPT}"
echo "term     : train_terminator=${TRAIN_TERMINATOR}   dino: ${DINO_TOKENS_PATH}"
echo "output   : ${PT_OUTPUT_DIR}   lr/bs: ${LR}/${BATCH_SIZE}"
nvidia-smi || true

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}  (run configs/train_skillVLA/build_data first)" >&2
  exit 1
fi
if [ ! -e "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-1 checkpoint: ${STAGE1_CHECKPOINT_PATH}  (run configs/train_skillVLA/stage1 first)" >&2
  exit 1
fi
if [ "${TRAIN_TERMINATOR}" = "true" ] && [ ! -e "${DINO_TOKENS_PATH}" ]; then
  echo "train_terminator=true but DINO tokens missing: ${DINO_TOKENS_PATH}  (rebuild dataset)" >&2
  exit 1
fi

# Resume if a Stage-2 checkpoint exists; otherwise wipe a stale (incomplete) output dir.
CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found → resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "No checkpoint → removing stale ${PT_OUTPUT_DIR} and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

# Terminator co-training inputs: attach current-frame DINO tokens only when train_terminator=true.
TERM_ARGS=()
if [ "${TRAIN_TERMINATOR}" = "true" ]; then
  TERM_ARGS=(
    --policy.train_terminator=true
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}"
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}"
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}"
    --policy.skill_decoder_dino_tokens_path="${DINO_TOKENS_PATH}"
    --policy.skill_decoder_dino_output_key=skill_decoder_dino
  )
  # Dual (terminator_use_wrist) FSQ → also attach the wrist DINO tokens (emitter sets this only if built).
  if [ -n "${DINO_WRIST_TOKENS_PATH:-}" ]; then
    TERM_ARGS+=(
      --policy.skill_decoder_dino_wrist_tokens_path="${DINO_WRIST_TOKENS_PATH}"
      --policy.skill_decoder_dino_wrist_output_key=skill_decoder_dino_wrist
    )
  fi
fi

# pretrained_path=pi05 warm-starts the VLM; stage1_checkpoint_path warm-starts the action/cond side
# (the policy's from_pretrained loads both). vision/skill_vocab/state_cond_mode are read from Stage-1.
PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.skill_loss_weight="${SKILL_LOSS_WEIGHT}" \
    --policy.attend_language="${ATTEND_LANGUAGE}" \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --policy.vlm_dropout_p="${VLM_DROPOUT_P}" \
    --policy.freeze_vlm_vsa_expert="${FREEZE_VLM_VSA_EXPERT}" \
    --policy.freeze_vlm_vsa_cond="${FREEZE_VLM_VSA_COND}" \
    --policy.freeze_vlm_vsa_vlm="${FREEZE_VLM_VSA_VLM}" \
    --policy.freeze_vsa_expert="${FREEZE_VSA_EXPERT}" \
    --policy.freeze_vsa_cond="${FREEZE_VSA_COND}" \
    --policy.freeze_vsa_vlm="${FREEZE_VSA_VLM}" \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.freeze_vlm="${FREEZE_VLM}" \
    --policy.freeze_vlm_vision="${FREEZE_VLM_VISION}" \
    --policy.freeze_cond_encoder="${FREEZE_COND_ENCODER}" \
    --policy.freeze_action_expert="${FREEZE_ACTION_EXPERT}" \
    --policy.freeze_expert_vision="${FREEZE_EXPERT_VISION}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.optimizer_lr="${LR}" \
    --policy.expert_lr_scale="${EXPERT_LR_SCALE}" \
    --policy.cond_lr_scale="${COND_LR_SCALE}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${TERM_ARGS[@]+"${TERM_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Export the co-trained terminator per saved checkpoint (checkpoints/<step>/FSQ_ft.pt) for stage2_eval
# (the terminator weights live in each checkpoint's model.safetensors; reuses the FT export script).
if [ "${TRAIN_TERMINATOR}" = "true" ]; then
  for CKPT_DIR in "${PT_OUTPUT_DIR}"/checkpoints/*/; do
    [ -d "${CKPT_DIR}pretrained_model" ] || continue
    STEP="$(basename "${CKPT_DIR}")"
    OUT="${CKPT_DIR%/}/FSQ_ft.pt"
    [ -f "${OUT}" ] && continue
    echo "Exporting terminator @ ${STEP} → ${OUT}"
    "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/../../FT/src/export_ft_terminator.py" \
      --ft_run_dir="${PT_OUTPUT_DIR}" --checkpoint="${STEP}" \
      --base_fsq="${FSQ_CKPT}" --out="${OUT}" --skip_if_no_terminator
  done
fi

echo "DONE → ${PT_OUTPUT_DIR}"
