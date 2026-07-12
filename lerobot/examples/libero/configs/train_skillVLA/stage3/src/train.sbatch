#!/usr/bin/env bash
# SkillVLA STAGE 3 training: policy.type=skill_vla, pt_stage="skill".
# Warm-starts the WHOLE policy from a STAGE-2 checkpoint (pretrained_path → from_pretrained's is_stage2
# full-load) and trains ONLY the skill path (adapter ① + skill_reader + skill_head) on the skill loss,
# over the SAME dataset stage 2 trained on. Everything else frozen (vision included). Resumes from a
# stage-3 checkpoint if one exists.
#SBATCH --job-name=stage3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy → anchor on SLURM_SUBMIT_DIR (the stage3 dir).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE3_TRAIN_CONFIG:-${SUBMIT_DIR}/stage3_train_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Prefer the submit-time env snapshot (frozen resolved env) → the job never re-runs the emitter on a
# live/edited/deleted yaml at start. Fallback to a fresh resolve only when run directly (no snapshot).
if [ -n "${STAGE3_ENV_SNAPSHOT:-}" ] && [ -f "${STAGE3_ENV_SNAPSHOT}" ]; then
  source "${STAGE3_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage3_train_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START STAGE-3 (skill_vla, pt_stage=skill)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}  (inherited from the stage-2 run)"
echo "warm-start: ${STAGE2_CHECKPOINT_PATH}"
echo "stage1cfg : ${STAGE1_CHECKPOINT_PATH:-<none>}"
echo "output    : ${PT_OUTPUT_DIR}   lr/bs: ${LR}/${BATCH_SIZE}  (① lr ×${LORA_LR_SCALE})"
nvidia-smi || true

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -e "${STAGE2_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-2 checkpoint: ${STAGE2_CHECKPOINT_PATH}  (train stage2 first)" >&2
  exit 1
fi
if [ "${TRAIN_TERMINATOR}" = "true" ] && [ ! -e "${DINO_TOKENS_PATH}" ]; then
  echo "train_terminator=true but DINO tokens missing: ${DINO_TOKENS_PATH}" >&2
  exit 1
fi

# Resume if a stage-3 checkpoint exists; otherwise wipe a stale (incomplete) output dir.
CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found → resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "No checkpoint → removing stale ${PT_OUTPUT_DIR} and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

# Transition pack — mode a's training set (segment-level skill-start pairs; the VLM's deployment
# distribution) AND the future FT-replay buffer. Lazy-build if missing (cheap, one-time) — built even
# in grounding mode (b trains on the FULL dataset, but FT replay still wants the pack).
if [ ! -f "${TRANSITION_PACK}" ]; then
  echo "Transition pack missing → building ${TRANSITION_PACK}"
  PYTHONPATH="${LEROBOT_ROOT}/src" \
    "${PROJECT_ROOT}/.venv/bin/python" "${SUBMIT_DIR}/../build_data/src/build_transition_pack.py" \
    --skillvla_dir="${SKILLVLA_DATASET_DIR}" --out="${TRANSITION_PACK}"
fi
# mode a → segment-level pack dataset; mode b (grounding) → the full frame-level dataset.
PACK_ARGS=()
if [ "${SKILL_ACTION_GROUNDING}" != "true" ]; then
  PACK_ARGS=(--policy.transition_packs="${TRANSITION_PACK}")
fi

# Terminator co-training inputs (usually off at stage 3 — stage 2 already adapted it on this dataset).
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
  if [ -n "${DINO_WRIST_TOKENS_PATH:-}" ]; then
    TERM_ARGS+=(
      --policy.skill_decoder_dino_wrist_tokens_path="${DINO_WRIST_TOKENS_PATH}"
      --policy.skill_decoder_dino_wrist_output_key=skill_decoder_dino_wrist
    )
  fi
fi

# pretrained_path = the STAGE-2 checkpoint (full-loaded); stage1_checkpoint_path supplies the
# architecture config only (no weights reloaded). ① is injected fresh (no keys in the checkpoint →
# B=0 no-op start); ②③ re-injected to match the checkpoint keys and stay frozen.
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
    --policy.skill_reader_all_layers="${SKILL_READER_ALL_LAYERS}" \
    --policy.attend_language="${ATTEND_LANGUAGE}" \
    --policy.attend_image="${ATTEND_IMAGE}" \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --track_param_drift="${TRACK_PARAM_DRIFT}" \
    --policy.pt_stage="${PT_STAGE}" \
    --policy.skill_action_grounding="${SKILL_ACTION_GROUNDING}" \
    ${PACK_ARGS[@]+"${PACK_ARGS[@]}"} \
    --policy.lora_skill="${LORA_SKILL}" \
    --policy.lora_cond_vlm="${LORA_COND_VLM}" \
    --policy.lora_cond_bridge="${LORA_COND_BRIDGE}" \
    --policy.lora_rank="${LORA_RANK}" \
    --policy.lora_alpha="${LORA_ALPHA}" \
    --policy.lora_dropout="${LORA_DROPOUT}" \
    --policy.lora_targets="${LORA_TARGETS}" \
    --policy.lora_lr_scale="${LORA_LR_SCALE}" \
    --policy.fsq_path="${FSQ_CKPT}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.optimizer_lr="${LR}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${TERM_ARGS[@]+"${TERM_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

# Export the co-trained terminator per saved checkpoint (only when co-trained at THIS stage).
if [ "${TRAIN_TERMINATOR}" = "true" ]; then
  for CKPT_DIR in "${PT_OUTPUT_DIR}"/checkpoints/*/; do
    [ -d "${CKPT_DIR}pretrained_model" ] || continue
    STEP="$(basename "${CKPT_DIR}")"
    OUT="${CKPT_DIR}FSQ_ft.pt"
    [ -f "${OUT}" ] && continue
    "${PROJECT_ROOT}/.venv/bin/python" "${SUBMIT_DIR}/../FT/src/export_ft_terminator.py" \
      --ft_run_dir="${PT_OUTPUT_DIR}" --checkpoint="${STEP}" \
      --base_fsq="${FSQ_CKPT}" --out="${OUT}" --skip_if_no_terminator \
      || echo "terminator export failed for ${STEP} (non-fatal)"
  done
fi

echo "DONE → ${PT_OUTPUT_DIR}"
