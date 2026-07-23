#!/usr/bin/env bash
# SkillVLA Stage-0: pi05 expert + VSA base path + bounded VLM residual.
#SBATCH --job-name=stage0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE0_TRAIN_CONFIG:-${SUBMIT_DIR}/stage0_train_config.yaml}"
mkdir -p "${SUBMIT_DIR}/logs"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if [ -n "${STAGE0_ENV_SNAPSHOT:-}" ] && [ -f "${STAGE0_ENV_SNAPSHOT}" ]; then
  source "${STAGE0_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage0_train_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

echo "START STAGE-0 (VSA base + VLM residual)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "VLM       : ${PI_BASE} (freeze llm=${STAGE0_FREEZE_VLM_LLM}, vision=${STAGE0_FREEZE_VLM_VISION})"
echo "expert    : pi05_base + state/skill conditioning (${STATE_COND_MODE})"
echo "cond      : ${VISION_BACKBONE}/${COND_ENCODER_VARIANT}"
echo "residual  : heads=${STAGE0_VLM_RESIDUAL_HEADS}, alpha=${STAGE0_VLM_RESIDUAL_INIT_ALPHA} in [${STAGE0_VLM_RESIDUAL_ALPHA_MIN},${STAGE0_VLM_RESIDUAL_ALPHA_MAX}], zero_out=${STAGE0_VLM_RESIDUAL_ZERO_INIT_OUTPUT}"
echo "skill pred: train=${STAGE0_TRAIN_SKILL_PREDICTOR}, weight=${STAGE0_SKILL_PREDICTOR_WEIGHT}, all_layers=${STAGE0_SKILL_PREDICTOR_ALL_LAYERS}"
echo "loss      : cond=${STAGE0_CONDITIONAL_LOSS_WEIGHT}, uncond=${STAGE0_UNCONDITIONAL_LOSS_WEIGHT}"
echo "batching  : batch=${BATCH_SIZE}, same-skill=${SAME_SKILL_BATCH_ENABLED}, grouped=${SAME_SKILL_BATCH_FRACTION}, progress-temp=${SAME_SKILL_PROGRESS_TEMPERATURE}"
echo "grad ckpt : ${GRADIENT_CHECKPOINTING}"
echo "output    : ${PT_OUTPUT_DIR}"
nvidia-smi || true

for required in "${SKILLVLA_DATASET_DIR}" "${PI_BASE}" "${TOKENIZER_PATH}" "${FSQ_CKPT}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing Stage-0 prerequisite: ${required}" >&2
    exit 1
  fi
done

CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found -> resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.stage1_checkpoint_path="" \
    --policy.pt_stage=stage0 \
    --policy.stage0_expert_source=pi05_base \
    --policy.stage0_cond_state_adarms="${COND_STATE_ADARMS}" \
    --policy.stage0_vlm_residual="${STAGE0_VLM_RESIDUAL}" \
    --policy.stage0_vlm_residual_heads="${STAGE0_VLM_RESIDUAL_HEADS}" \
    --policy.stage0_vlm_residual_dropout="${STAGE0_VLM_RESIDUAL_DROPOUT}" \
    --policy.stage0_vlm_residual_alpha_min="${STAGE0_VLM_RESIDUAL_ALPHA_MIN}" \
    --policy.stage0_vlm_residual_alpha_max="${STAGE0_VLM_RESIDUAL_ALPHA_MAX}" \
    --policy.stage0_vlm_residual_init_alpha="${STAGE0_VLM_RESIDUAL_INIT_ALPHA}" \
    --policy.stage0_vlm_residual_zero_init_output="${STAGE0_VLM_RESIDUAL_ZERO_INIT_OUTPUT}" \
    --policy.stage0_train_skill_predictor="${STAGE0_TRAIN_SKILL_PREDICTOR}" \
    --policy.stage0_skill_predictor_weight="${STAGE0_SKILL_PREDICTOR_WEIGHT}" \
    --policy.stage0_skill_predictor_lr_scale="${STAGE0_SKILL_PREDICTOR_LR_SCALE}" \
    --policy.stage0_skill_predictor_all_layers="${STAGE0_SKILL_PREDICTOR_ALL_LAYERS}" \
    --policy.stage0_skill_predictor_detach_vlm="${STAGE0_SKILL_PREDICTOR_DETACH_VLM}" \
    --policy.skill_reader_all_layers="${SKILL_READER_ALL_LAYERS}" \
    --policy.stage0_conditional_loss_weight="${STAGE0_CONDITIONAL_LOSS_WEIGHT}" \
    --policy.stage0_unconditional_loss_weight="${STAGE0_UNCONDITIONAL_LOSS_WEIGHT}" \
    --policy.stage0_uncond_skill_start_loss_weight="${STAGE0_UNCOND_SKILL_START_LOSS_WEIGHT}" \
    --policy.stage0_uncond_skill_end_loss_weight="${STAGE0_UNCOND_SKILL_END_LOSS_WEIGHT}" \
    --policy.same_skill_batch_enabled="${SAME_SKILL_BATCH_ENABLED}" \
    --policy.same_skill_batch_fraction="${SAME_SKILL_BATCH_FRACTION}" \
    --policy.same_skill_progress_temperature="${SAME_SKILL_PROGRESS_TEMPERATURE}" \
    --policy.stage0_freeze_vlm_llm="${STAGE0_FREEZE_VLM_LLM}" \
    --policy.stage0_freeze_vlm_vision="${STAGE0_FREEZE_VLM_VISION}" \
    --policy.stage0_freeze_cond="${STAGE0_FREEZE_COND}" \
    --policy.stage0_freeze_cond_vision="${STAGE0_FREEZE_COND_VISION}" \
    --policy.stage0_freeze_expert="${STAGE0_FREEZE_EXPERT}" \
    --policy.stage0_freeze_skill_reader="${STAGE0_FREEZE_SKILL_READER}" \
    --policy.stage0_freeze_skill_head="${STAGE0_FREEZE_SKILL_HEAD}" \
    --policy.cond_skill_source=gt \
    --policy.s1_vision_backbone="${VISION_BACKBONE}" \
    --policy.s1_cond_encoder_variant="${COND_ENCODER_VARIANT}" \
    --policy.s1_state_cond_mode="${STATE_COND_MODE}" \
    --policy.s1_dino_model_path="${DINO_MODEL_PATH}" \
    --policy.s1_dino_image_size="${DINO_IMAGE_SIZE}" \
    --policy.s1_siglip_image_size="${SIGLIP_IMAGE_SIZE}" \
    --policy.action_expert_variant="${ACTION_EXPERT_VARIANT}" \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${CHUNK_SIZE}" \
    --policy.min_period="${MIN_PERIOD}" \
    --policy.max_period="${MAX_PERIOD}" \
    --policy.time_sampling_beta_alpha="${TIME_SAMPLING_BETA_ALPHA}" \
    --policy.time_sampling_beta_beta="${TIME_SAMPLING_BETA_BETA}" \
    --policy.time_sampling_scale="${TIME_SAMPLING_SCALE}" \
    --policy.time_sampling_offset="${TIME_SAMPLING_OFFSET}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.attend_image="${ATTEND_IMAGE}" \
    --policy.attend_language="${ATTEND_LANGUAGE}" \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --policy.lora_skill=false \
    --policy.vlm_lora=false \
    --policy.cond_lora=false \
    --policy.stage0_expert_lora=false \
    --policy.lang_bridge=false \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.terminator_freeze_vision_encoder="${TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.terminator_dino_model_path="${DINO_MODEL_PATH}" \
    --policy.optimizer_lr="${LR}" \
    --policy.cond_lr_scale="${COND_LR_SCALE}" \
    --policy.expert_lr_scale="${EXPERT_LR_SCALE}" \
    --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --track_param_drift="${TRACK_PARAM_DRIFT}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
