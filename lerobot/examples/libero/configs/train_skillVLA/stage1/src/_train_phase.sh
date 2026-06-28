# Shared by train_joint.sbatch / train_staged.sbatch. SOURCE this AFTER the emitter `eval` (it relies on
# the exported settings). Sets up the env and defines train_phase() — one lerobot-train run = one fixed
# PHASE. Per-phase knobs are positional args; everything else comes from the shared env exports.

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}  (run configs/train_skillVLA/build_data first)" >&2
  exit 1
fi
if [ "${TRAIN_TERMINATOR}" = "true" ]; then   # terminator inputs are auto-derived build_data products
  for _f in "${FSQ_PATH}" "${SKILL_DECODER_DINO_TOKENS_PATH}"; do
    if [ ! -e "${_f}" ]; then
      echo "train_terminator=true but a terminator input is missing: ${_f}" >&2
      echo "  → expected a build_data product under the run dir (FSQ.pt / dino.npz). Rebuild the dataset," >&2
      echo "    or set train_terminator: false in the yaml." >&2
      exit 1
    fi
  done
fi
nvidia-smi || true

# train_phase OUT STEPS USE_CONNECTOR ACTION_WEIGHTING FREEZE_EXPERT_VISION BOUNDARY_MODE PRETRAINED LOSS_MODE
train_phase() {
  local OUT="$1" STEPS_="$2" USE_CONN="$3" ACT_W="$4" FREEZE_EV="$5" BOUND="$6" PRETRAINED="$7" LMODE="$8"

  local EXTRA=()
  [ -n "${PRETRAINED}" ] && EXTRA+=(--policy.pretrained_path="${PRETRAINED}")   # pi05 (joint/1-1) or 1-1 ckpt (1-2)
  [ -n "${DINO_LR}" ] && EXTRA+=(--policy.dino_lr="${DINO_LR}")
  [ -n "${SIGLIP_LR}" ] && EXTRA+=(--policy.siglip_lr="${SIGLIP_LR}")
  [ -n "${COND_ENCODER_VARIANT}" ] && EXTRA+=(--policy.cond_encoder_variant="${COND_ENCODER_VARIANT}")
  [ -n "${FSQ_PATH}" ] && EXTRA+=(--policy.fsq_path="${FSQ_PATH}")
  [ -n "${SKILL_DECODER_DINO_TOKENS_PATH}" ] && EXTRA+=(--policy.skill_decoder_dino_tokens_path="${SKILL_DECODER_DINO_TOKENS_PATH}")
  [ -n "${SKILL_DECODER_DINO_CACHE_PATH}" ] && EXTRA+=(--policy.skill_decoder_dino_cache_path="${SKILL_DECODER_DINO_CACHE_PATH}")

  # Resume from THIS phase's own checkpoint if present; else wipe a stale dir and start (warm-)fresh.
  local CFG_JSON="${OUT}/checkpoints/last/pretrained_model/train_config.json"
  local RESUME=()
  if [ -f "${CFG_JSON}" ]; then
    echo "Checkpoint found → resuming ${OUT}"
    RESUME=(--resume=true --config_path="${CFG_JSON}")
  elif [ -d "${OUT}" ]; then
    echo "No checkpoint → removing stale ${OUT}."
    rm -rf "${OUT}"
  fi

  echo "── train_phase → ${OUT}  (steps=${STEPS_} use_connector=${USE_CONN} weighting=${ACT_W} freeze_ev=${FREEZE_EV} boundary=${BOUND} init=${PRETRAINED:-scratch}) ──"
  PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
    "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
      --dataset.repo_id="${REPO_ID}" \
      --dataset.root="${SKILLVLA_DATASET_DIR}" \
      --dataset.video_backend=pyav \
      --policy.type=skill_expert \
      --output_dir="${OUT}" \
      --job_name="${PT_RUN_NAME}" \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.state_cond_mode="${STATE_COND_MODE}" \
      --policy.vision_backbone="${VISION_BACKBONE}" \
      --policy.dino_model_path="${DINO_MODEL_PATH}" \
      --policy.freeze_dino="${FREEZE_DINO}" \
      --policy.freeze_siglip="${FREEZE_SIGLIP}" \
      --policy.siglip_image_size="${SIGLIP_IMAGE_SIZE}" \
      --policy.skill_vocab_size="${SKILL_VOCAB_SIZE}" \
      --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
      --policy.chunk_size="${CHUNK_SIZE}" \
      --policy.n_action_steps="${N_ACTION_STEPS}" \
      --policy.skill_end_loss_weight="${SKILL_END_LOSS_WEIGHT}" \
      --policy.use_connector="${USE_CONN}" \
      --policy.connector_dino_model_path="${CONNECTOR_DINO_MODEL_PATH}" \
      --policy.connector_dino_image_size="${CONNECTOR_DINO_IMAGE_SIZE}" \
      --policy.connector_width="${CONNECTOR_WIDTH}" \
      --policy.connector_depth="${CONNECTOR_DEPTH}" \
      --policy.connector_n_heads="${CONNECTOR_N_HEADS}" \
      --policy.connector_n_latents="${CONNECTOR_N_LATENTS}" \
      --policy.connector_z_dim="${CONNECTOR_Z_DIM}" \
      --policy.connector_free_bits="${CONNECTOR_FREE_BITS}" \
      --policy.connector_kl_weight="${CONNECTOR_KL_WEIGHT}" \
      --policy.connector_z_consistency_weight="${CONNECTOR_Z_CONSISTENCY_WEIGHT}" \
      --policy.z_ablation_every="${Z_ABLATION_EVERY}" \
      --policy.loss_mode="${LMODE}" \
      --policy.action_weighting="${ACT_W}" \
      --policy.freeze_expert_vision="${FREEZE_EV}" \
      --policy.gate_prob="${GATE_PROB}" \
      --policy.boundary_mode="${BOUND}" \
      --policy.train_terminator="${TRAIN_TERMINATOR}" \
      --policy.terminator_end_target_sigma="${TERMINATOR_END_TARGET_SIGMA}" \
      --policy.terminator_end_pos_weight="${TERMINATOR_END_POS_WEIGHT}" \
      --policy.terminator_lr_scale="${TERMINATOR_LR_SCALE}" \
      --policy.skill_decoder_dino_output_key="${SKILL_DECODER_DINO_OUTPUT_KEY}" \
      --policy.skill_decoder_dino_build_cache="${SKILL_DECODER_DINO_BUILD_CACHE}" \
      --batch_size="${BATCH_SIZE}" \
      --num_workers="${NUM_WORKERS}" \
      --steps="${STEPS_}" \
      --save_freq="${SAVE_FREQ}" \
      --policy.optimizer_lr="${LR}" \
      --wandb.enable="${WANDB_ENABLE}" \
      --wandb.project="${WANDB_PROJECT}" \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      ${RESUME[@]+"${RESUME[@]}"}
}
