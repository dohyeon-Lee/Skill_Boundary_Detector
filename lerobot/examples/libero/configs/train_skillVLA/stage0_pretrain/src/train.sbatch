#!/usr/bin/env bash
# SkillVLA Stage-0-pretrain: direct FSQ motor + autoregressive pretrained VLM.
#SBATCH --job-name=s0pre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${STAGE0_PRETRAIN_CONFIG:-${SUBMIT_DIR}/stage0_pretrain_config.yaml}"
mkdir -p "${SUBMIT_DIR}/logs"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if [ -n "${STAGE0_PRETRAIN_ENV_SNAPSHOT:-}" ] && [ -f "${STAGE0_PRETRAIN_ENV_SNAPSHOT}" ]; then
  source "${STAGE0_PRETRAIN_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage0_pretrain_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

echo "START STAGE-0-PRETRAIN (AR VLM + direct FSQ motor)"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "VLM       : ${PRETRAIN_CHECKPOINT_PATH} (${PRETRAIN_TRAINING_MODE})"
echo "expert    : ${FSQ_CKPT} (trainability follows the A/B scopes below)"
echo "cond      : ${VISION_BACKBONE}/${COND_ENCODER_VARIANT} (trainability follows the A/B scopes below)"
echo "A/B       : B=${STAGE0_VLM_SEVERED_PROB}, wrong-language=${WRONG_LANGUAGE_WEIGHT}"
echo "A scope   : drop_vlm=${STAGE0_A_DROP_VLM}, train=${STAGE0_A_TRAIN_COMPONENTS}"
echo "B scope   : drop_vlm=${STAGE0_B_DROP_VLM}, train=${STAGE0_B_TRAIN_COMPONENTS}"
echo "token read: image=${ATTEND_IMAGE}, language=${ATTEND_LANGUAGE}, skill=${ATTEND_SKILL}, fast=${ATTEND_FAST}"
echo "AR        : transition randomization=${TRANSITION_RANDOMIZATION}, FAST=${AR_FAST_LOSS}"
echo "output    : ${PT_OUTPUT_DIR}"
nvidia-smi || true

for required in "${SKILLVLA_DATASET_DIR}" "${PI_BASE}" "${TOKENIZER_PATH}" "${FSQ_CKPT}" \
                "${PRETRAIN_CHECKPOINT_PATH}" "${TRANSITION_PACK}" "${PRETRAIN_TARGET_PACK}"; do
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
    --policy.type=skill_vla_stage0_pretrain \
    --output_dir="${PT_OUTPUT_DIR}" \
    --job_name="${PT_RUN_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.pretrain_checkpoint_path="${PRETRAIN_CHECKPOINT_PATH}" \
    --policy.pretrain_training_mode="${PRETRAIN_TRAINING_MODE}" \
    --policy.pretrain_lora_targets="${PRETRAIN_LORA_TARGETS}" \
    --policy.pretrain_lora_rank="${PRETRAIN_LORA_RANK}" \
    --policy.pretrain_lora_alpha="${PRETRAIN_LORA_ALPHA}" \
    --policy.pretrain_lora_dropout="${PRETRAIN_LORA_DROPOUT}" \
    --policy.skill_unused_start="${SKILL_UNUSED_START}" \
    --policy.fast_vocab_size="${FAST_VOCAB_SIZE}" \
    --policy.fast_skip_tokens="${FAST_SKIP_TOKENS}" \
    --policy.max_action_tokens="${MAX_ACTION_TOKENS}" \
    --policy.transition_packs="${TRANSITION_PACK}" \
    --policy.pretrain_target_packs="${PRETRAIN_TARGET_PACK}" \
    --policy.transition_randomization="${TRANSITION_RANDOMIZATION}" \
    --policy.attend_skill="${ATTEND_SKILL}" \
    --policy.attend_fast="${ATTEND_FAST}" \
    --policy.ar_fast_loss="${AR_FAST_LOSS}" \
    --policy.ar_batch_size="${AR_BATCH_SIZE}" \
    --policy.ar_skill_loss_weight="${AR_SKILL_LOSS_WEIGHT}" \
    --policy.ar_fast_loss_weight="${AR_FAST_LOSS_WEIGHT}" \
    --policy.ar_structure_loss_weight="${AR_STRUCTURE_LOSS_WEIGHT}" \
    --policy.tokenizer_path="${TOKENIZER_PATH}" \
    --policy.stage1_checkpoint_path="" \
    --policy.pt_stage=stage0 \
    --policy.stage0_expert_source="${EXPERT_SOURCE}" \
    --policy.stage0_cond_state_adarms="${COND_STATE_ADARMS}" \
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
    --policy.num_reader_tokens="${NUM_READER_TOKENS}" \
    --policy.reader_depth="${READER_DEPTH}" \
    --policy.reader_heads="${READER_HEADS}" \
    --policy.skill_reader_all_layers="${SKILL_READER_ALL_LAYERS}" \
    --policy.reader_attend_image="${READER_ATTEND_IMAGE}" \
    --policy.reader_attend_language="${READER_ATTEND_LANGUAGE}" \
    --policy.vlm_cond="${VLM_COND}" \
    --policy.cond_expert="${COND_EXPERT}" \
    --policy.vlm_expert="${VLM_EXPERT}" \
    --policy.lora_skill=false \
    --policy.vlm_lora="${VLM_LORA}" \
    --policy.cond_lora="${COND_LORA}" \
    --policy.lora_targets="${VLM_LORA_TARGETS}" \
    --policy.lora_rank="${VLM_LORA_RANK}" \
    --policy.lora_alpha="${VLM_LORA_ALPHA}" \
    --policy.lora_dropout="${VLM_LORA_DROPOUT}" \
    --policy.lora_lr_scale="${VLM_LORA_LR_SCALE}" \
    --policy.stage0_expert_lora="${STAGE0_EXPERT_LORA}" \
    --policy.stage0_expert_lora_targets="${EXPERT_LORA_TARGETS}" \
    --policy.stage0_expert_lora_rank="${EXPERT_LORA_RANK}" \
    --policy.stage0_expert_lora_alpha="${EXPERT_LORA_ALPHA}" \
    --policy.stage0_expert_lora_dropout="${EXPERT_LORA_DROPOUT}" \
    --policy.stage0_expert_lora_lr_scale="${EXPERT_LORA_LR_SCALE}" \
    --policy.lang_bridge="${LANG_BRIDGE}" \
    --policy.lang_bridge_rank="${LANG_BRIDGE_RANK}" \
    --policy.lang_bridge_lr_scale="${LANG_BRIDGE_LR_SCALE}" \
    --policy.stage0_vlm_severed_prob="${STAGE0_VLM_SEVERED_PROB}" \
    --policy.stage0_a_drop_vlm="${STAGE0_A_DROP_VLM}" \
    --policy.stage0_b_drop_vlm="${STAGE0_B_DROP_VLM}" \
    --policy.stage0_a_train_components="${STAGE0_A_TRAIN_COMPONENTS}" \
    --policy.stage0_b_train_components="${STAGE0_B_TRAIN_COMPONENTS}" \
    --policy.stage0_wrong_language_weight="${WRONG_LANGUAGE_WEIGHT}" \
    --policy.stage0_wrong_language_margin="${WRONG_LANGUAGE_MARGIN}" \
    --policy.stage0_a_skill_start_loss_weight="${A_SKILL_START_LOSS_WEIGHT}" \
    --policy.stage0_a_skill_end_loss_weight="${A_SKILL_END_LOSS_WEIGHT}" \
    --policy.stage0_b_skill_start_loss_weight="${B_SKILL_START_LOSS_WEIGHT}" \
    --policy.stage0_b_skill_end_loss_weight="${B_SKILL_END_LOSS_WEIGHT}" \
    --policy.severed_hold_target=true \
    --policy.fsq_path="${FSQ_CKPT}" \
    --policy.train_terminator="${TRAIN_TERMINATOR}" \
    --policy.terminator_freeze_vision_encoder="${TERMINATOR_FREEZE_VISION_ENCODER}" \
    --policy.terminator_dino_model_path="${DINO_MODEL_PATH}" \
    --policy.optimizer_lr="${LR}" \
    --policy.cond_lr_scale="${COND_LR_SCALE}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.scheduler_decay_lr="$(${BOOTSTRAP_PYTHON} -c "print(${LR} * 0.1)")" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --track_param_drift="${TRACK_PARAM_DRIFT}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
