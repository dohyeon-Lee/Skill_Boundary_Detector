#!/usr/bin/env bash
# SkillVLA STAGE 3 training: policy.type=skill_vla, pt_stage="skill".
# Full-loads a frozen Stage-0/2 motor and trains only the skill path
# (skill adapter + skill_reader + skill_head). pi05_base swaps only the predictor VLM.
#SBATCH --job-name=stage3
#SBATCH --gres=gpu:1
#SBATCH --requeue
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
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

echo "START STAGE-3 (skill_vla, pt_stage=skill)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "parent    : ${PARENT_STAGE} / ${PARENT_RUN_NAME} / ${PARENT_CHECKPOINT}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "warm-start: ${PARENT_CHECKPOINT_PATH}"
echo "predictor : ${SKILL_PREDICTOR_VLM_PATH:-<parent VLM>}"
echo "FSQ       : ${FSQ_CKPT} (${FSQ_SOURCE})"
echo "stage1cfg : ${STAGE1_CHECKPOINT_PATH:-<none>}"
echo "output    : ${PT_OUTPUT_DIR}   lr/bs: ${LR}/${BATCH_SIZE}  (① lr ×${LORA_LR_SCALE})"
nvidia-smi || true

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -e "${PARENT_CHECKPOINT_PATH}" ]; then
  echo "Missing ${PARENT_STAGE} checkpoint: ${PARENT_CHECKPOINT_PATH}" >&2
  exit 1
fi
if [ -n "${SKILL_PREDICTOR_VLM_PATH}" ] && [ ! -e "${SKILL_PREDICTOR_VLM_PATH}" ]; then
  echo "Missing predictor VLM checkpoint: ${SKILL_PREDICTOR_VLM_PATH}" >&2
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
# in grounding mode (b trains on the FULL dataset, but FT replay still wants the pack). Schema v3 stores
# the raw start-state window and jitter distribution; older packs are rebuilt in place.
if [ -f "${TRANSITION_PACK}" ] && "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import numpy as np,sys; z=np.load(sys.argv[1]); assert "start_state" in z and "jitter_distribution" in z and int(z["schema_version"]) >= 3' \
    "${TRANSITION_PACK}"; then
  echo "Transition pack schema OK: ${TRANSITION_PACK}"
else
  echo "Transition pack missing/old → building schema v3 ${TRANSITION_PACK}"
  PYTHONPATH="${LEROBOT_ROOT}/src" \
    "${PROJECT_ROOT}/.venv/bin/python" "${SUBMIT_DIR}/../build_data/src/build_transition_pack.py" \
    --skillvla_dir="${SKILLVLA_DATASET_DIR}" --out="${TRANSITION_PACK}"
fi
# mode a → segment-level pack dataset; mode b (grounding) → the full frame-level dataset.
PACK_ARGS=()
if [ "${SKILL_ACTION_GROUNDING}" != "true" ]; then
  PACK_ARGS=(--policy.transition_packs="${TRANSITION_PACK}")
fi

# Terminator co-training inputs. FSQ_CKPT already prefers the parent's exported co-trained terminator.
TERM_ARGS=()
if [ "${TRAIN_TERMINATOR}" = "true" ]; then
  TERM_ARGS=(
    --policy.train_terminator=true
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}"
    --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}"
    --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}"
  )
fi

# pretrained_path always full-loads the frozen motor parent. The optional predictor-only VLM is loaded
# separately; its fresh skill adapter starts as a zero-output no-op and is the only trained adapter.
READER_ARGS=()
[ -n "${READER_ATTEND_IMAGE:-}" ] && READER_ARGS+=(--policy.reader_attend_image="${READER_ATTEND_IMAGE}")
[ -n "${READER_ATTEND_LANGUAGE:-}" ] && READER_ARGS+=(--policy.reader_attend_language="${READER_ATTEND_LANGUAGE}")
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
    --policy.pretrained_path="${PARENT_CHECKPOINT_PATH}" \
    --policy.stage3_parent_stage="${STAGE3_PARENT_STAGE}" \
    --policy.skill_predictor_vlm_path="${SKILL_PREDICTOR_VLM_PATH}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
    --policy.paligemma_variant="${PALIGEMMA_VARIANT}" \
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.s1_vision_backbone="${S1_VISION_BACKBONE}" \
    --policy.s1_state_cond_mode="${S1_STATE_COND_MODE}" \
    --policy.s1_cond_encoder_variant="${S1_COND_ENCODER_VARIANT}" \
    --policy.s1_dino_model_path="${S1_DINO_MODEL_PATH}" \
    --policy.s1_dino_image_size="${S1_DINO_IMAGE_SIZE}" \
    --policy.s1_siglip_image_size="${S1_SIGLIP_IMAGE_SIZE}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.min_period="${MIN_PERIOD}" \
    --policy.max_period="${MAX_PERIOD}" \
    --policy.time_sampling_beta_alpha="${TIME_SAMPLING_BETA_ALPHA}" \
    --policy.time_sampling_beta_beta="${TIME_SAMPLING_BETA_BETA}" \
    --policy.time_sampling_scale="${TIME_SAMPLING_SCALE}" \
    --policy.time_sampling_offset="${TIME_SAMPLING_OFFSET}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.skill_loss_weight="${SKILL_LOSS_WEIGHT}" \
    --policy.skill_deadzone_frac="${SKILL_DEADZONE_FRAC}" \
    --policy.num_reader_tokens="${NUM_READER_TOKENS}" \
    --policy.reader_depth="${READER_DEPTH}" \
    --policy.reader_heads="${READER_HEADS}" \
    --policy.skill_reader_all_layers="${SKILL_READER_ALL_LAYERS}" \
    --policy.attend_language="${ATTEND_LANGUAGE}" \
    --policy.attend_image="${ATTEND_IMAGE}" \
    ${READER_ARGS[@]+"${READER_ARGS[@]}"} \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --track_param_drift="${TRACK_PARAM_DRIFT}" \
    --policy.pt_stage="${PT_STAGE}" \
    --policy.skill_action_grounding="${SKILL_ACTION_GROUNDING}" \
    ${PACK_ARGS[@]+"${PACK_ARGS[@]}"} \
    --policy.lora_skill="${LORA_SKILL}" \
    --policy.vlm_lora="${VLM_LORA}" \
    --policy.cond_lora="${COND_LORA}" \
    --policy.lora_rank="${LORA_RANK}" \
    --policy.lora_alpha="${LORA_ALPHA}" \
    --policy.lora_dropout="${LORA_DROPOUT}" \
    --policy.lora_targets="${LORA_TARGETS}" \
    --policy.lora_lr_scale="${LORA_LR_SCALE}" \
    --policy.lang_bridge="${LANG_BRIDGE}" \
    --policy.lang_bridge_rank="${LANG_BRIDGE_RANK}" \
    --policy.lang_bridge_lr_scale="${LANG_BRIDGE_LR_SCALE}" \
    --policy.stage0_expert_source="${STAGE0_EXPERT_SOURCE}" \
    --policy.stage0_cond_state_adarms="${STAGE0_COND_STATE_ADARMS}" \
    --policy.stage0_expert_lora="${STAGE0_EXPERT_LORA}" \
    --policy.stage0_expert_lora_targets="${STAGE0_EXPERT_LORA_TARGETS}" \
    --policy.stage0_expert_lora_rank="${STAGE0_EXPERT_LORA_RANK}" \
    --policy.stage0_expert_lora_alpha="${STAGE0_EXPERT_LORA_ALPHA}" \
    --policy.stage0_expert_lora_dropout="${STAGE0_EXPERT_LORA_DROPOUT}" \
    --policy.stage0_expert_lora_lr_scale="${STAGE0_EXPERT_LORA_LR_SCALE}" \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.terminator_dino_model_path="${TERMINATOR_DINO_MODEL_PATH}" \
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
