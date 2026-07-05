#!/usr/bin/env bash
# SkillVLA FINETUNING (FT): policy.type=skill_vla, warm-started from a Stage-2 checkpoint
# (pretrained_path → from_pretrained is_stage2 full-load) and trained on a NEW task. cond_skill_source
# =pred self-conditions the action prefix on the VLM's predicted skill; the FSQ terminator is co-trained
# on GT signals and exported to FSQ_ft.pt for eval. Resumes from an FT checkpoint if one exists.
#SBATCH --job-name=ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR (the FT dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FT_TRAIN_CONFIG:-${SUBMIT_DIR}/ft_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_train_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START SkillVLA FT (skill_vla)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "warm-start: ${STAGE2_CHECKPOINT_PATH}"
echo "stage1cfg : ${STAGE1_CHECKPOINT_PATH:-SCRATCH-parent (${S1_VISION_BACKBONE}/${S1_STATE_COND_MODE})}"
echo "FSQ(term) : ${FSQ_CKPT}   dino: ${DINO_TOKENS_PATH}"
echo "cond_skill: ${COND_SKILL_SOURCE}   train_terminator: ${TRAIN_TERMINATOR}"
echo "output    : ${PT_OUTPUT_DIR}   lr/bs: ${LR}/${BATCH_SIZE}"
nvidia-smi || true

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}  (run configs/train_skillVLA/build_data first)" >&2
  exit 1
fi
if [ ! -e "${STAGE2_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-2 checkpoint: ${STAGE2_CHECKPOINT_PATH}  (train Stage-2 first)" >&2
  exit 1
fi
# SCRATCH-parent: the forked Stage-2 was trained without Stage-1 → empty path is fine (the model
# synthesizes the Stage-1-side architecture from --policy.s1_vision_backbone/s1_state_cond_mode).
if [ -n "${STAGE1_CHECKPOINT_PATH}" ] && [ ! -e "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-1 checkpoint (architecture config): ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 1
fi
if [ "${TRAIN_TERMINATOR}" = "true" ] && [ ! -e "${DINO_TOKENS_PATH}" ]; then
  echo "train_terminator=true but DINO tokens missing: ${DINO_TOKENS_PATH}  (rebuild dataset)" >&2
  exit 1
fi

# Resume if an FT checkpoint exists; otherwise wipe a stale (incomplete) output dir.
CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found → resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "No checkpoint → removing stale ${PT_OUTPUT_DIR} and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

# Dropout schedule (optional): pass p_end/decay only when a schedule is configured, so a fixed-p or
# p=0 run's train_config.json stays byte-identical to pre-schedule runs (resume-safe).
SCHED_ARGS=()
if [ -n "${VLM_DROPOUT_P_END}" ]; then
  SCHED_ARGS=(
    --policy.vlm_dropout_p_end="${VLM_DROPOUT_P_END}"
    --policy.vlm_dropout_decay_steps="${VLM_DROPOUT_DECAY_STEPS}"
  )
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
fi

# pretrained_path = Stage-2 checkpoint (full-loaded); stage1_checkpoint_path supplies the architecture
# config only (no weights reloaded); fsq_path = the new task's FSQ (terminator warm-start + eval base).
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
    --policy.pretrained_path="${STAGE2_CHECKPOINT_PATH}" \
    --policy.stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
    --policy.s1_vision_backbone="${S1_VISION_BACKBONE}" \
    --policy.s1_state_cond_mode="${S1_STATE_COND_MODE}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.skill_loss_weight="${SKILL_LOSS_WEIGHT}" \
    --policy.skill_deadzone_frac="${SKILL_DEADZONE_FRAC}" \
    --policy.num_reader_tokens="${NUM_READER_TOKENS}" \
    --policy.reader_depth="${READER_DEPTH}" \
    --policy.reader_heads="${READER_HEADS}" \
    --policy.attend_language="${ATTEND_LANGUAGE}" \
    --policy.attend_image="${ATTEND_IMAGE}" \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.cond_skill_source="${COND_SKILL_SOURCE}" \
    --policy.freeze_vlm_vsa_expert="${FREEZE_VLM_VSA_EXPERT}" \
    --policy.freeze_vlm_vsa_cond="${FREEZE_VLM_VSA_COND}" \
    --policy.freeze_vlm_vsa_llm="${FREEZE_VLM_VSA_LLM}" \
    --policy.freeze_vlm_vsa_vlm_vision="${FREEZE_VLM_VSA_VLM_VISION}" \
    --policy.freeze_vsa_expert="${FREEZE_VSA_EXPERT}" \
    --policy.freeze_vsa_cond="${FREEZE_VSA_COND}" \
    --policy.freeze_vsa_llm="${FREEZE_VSA_LLM}" \
    --policy.freeze_vsa_vlm_vision="${FREEZE_VSA_VLM_VISION}" \
    --policy.freeze_skill_head="${FREEZE_SKILL_HEAD}" \
    --policy.vlm_dropout_p="${VLM_DROPOUT_P}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.optimizer_lr="${LR}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${SCHED_ARGS[@]+"${SCHED_ARGS[@]}"} \
    ${TERM_ARGS[@]+"${TERM_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Export the co-trained terminator into a standalone FSQ checkpoint PER saved checkpoint, co-located
# with it (checkpoints/<step>/FSQ_ft.pt), so FT_eval can use the terminator from the SAME timepoint as
# the policy it evaluates (the terminator weights live in each checkpoint's model.safetensors).
if [ "${TRAIN_TERMINATOR}" = "true" ]; then
  for CKPT_DIR in "${PT_OUTPUT_DIR}"/checkpoints/*/; do
    [ -d "${CKPT_DIR}pretrained_model" ] || continue
    STEP="$(basename "${CKPT_DIR}")"
    OUT="${CKPT_DIR%/}/FSQ_ft.pt"
    [ -f "${OUT}" ] && continue
    echo "Exporting terminator @ ${STEP} → ${OUT}"
    "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/export_ft_terminator.py" \
      --ft_run_dir="${PT_OUTPUT_DIR}" --checkpoint="${STEP}" \
      --base_fsq="${FSQ_CKPT}" --out="${OUT}" --skip_if_no_terminator
  done
fi

echo "DONE → ${PT_OUTPUT_DIR}"
