# Shared by train_staged_1 / train_staged_2 / train_single. SOURCE this AFTER the emitter `eval`
# (it relies on the exported settings). Sets up the env and defines train_phase() — one lerobot-train
# run = one PHASE. Per-phase knobs are positional args; everything else comes from the shared env.

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

# train_phase OUT NAME STEPS USE_ORACLE FREEZE_VSA_BASE PRETRAINED
#   OUT             output dir for this phase
#   USE_ORACLE      true (1-2 / single) → Oracle + r-slot=r ; false (1-1) → r-slot=null only
#   FREEZE_VSA_BASE true (staged 1-2) → freeze vision+skill_proj+null ; false (1-1 / single)
#   PRETRAINED      pi05 (1-1 / single) or a 1-1 checkpoint (staged 1-2)
train_phase() {
  local OUT="$1" NAME="$2" STEPS_="$3" USE_ORACLE="$4" FREEZE_VSA="$5" PRETRAINED="$6"

  local EXTRA=()
  [ -n "${PRETRAINED}" ] && EXTRA+=(--policy.pretrained_path="${PRETRAINED}")   # pi05 (1-1/single) or 1-1 ckpt (1-2)
  [ -n "${DINO_LR}" ] && EXTRA+=(--policy.dino_lr="${DINO_LR}")
  [ -n "${SIGLIP_LR}" ] && EXTRA+=(--policy.siglip_lr="${SIGLIP_LR}")
  [ -n "${COND_ENCODER_VARIANT}" ] && EXTRA+=(--policy.cond_encoder_variant="${COND_ENCODER_VARIANT}")
  [ -n "${FSQ_PATH}" ] && EXTRA+=(--policy.fsq_path="${FSQ_PATH}")
  [ -n "${SKILL_DECODER_DINO_TOKENS_PATH}" ] && EXTRA+=(--policy.skill_decoder_dino_tokens_path="${SKILL_DECODER_DINO_TOKENS_PATH}")
  [ -n "${SKILL_DECODER_DINO_WRIST_TOKENS_PATH:-}" ] && EXTRA+=(--policy.skill_decoder_dino_wrist_tokens_path="${SKILL_DECODER_DINO_WRIST_TOKENS_PATH}")
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

  echo "── train_phase → ${OUT}  (steps=${STEPS_} use_oracle=${USE_ORACLE} freeze_vsa_base=${FREEZE_VSA} init=${PRETRAINED:-scratch}) ──"
  PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
    "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
      --dataset.repo_id="${REPO_ID}" \
      --dataset.root="${SKILLVLA_DATASET_DIR}" \
      --dataset.video_backend=pyav \
      --policy.type=skill_expert \
      --output_dir="${OUT}" \
      --job_name="${NAME}" \
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
      --policy.use_oracle="${USE_ORACLE}" \
      --policy.oracle_resample_n="${ORACLE_RESAMPLE_N}" \
      --policy.oracle_spline_degree="${ORACLE_SPLINE_DEGREE}" \
      --policy.oracle_width="${ORACLE_WIDTH}" \
      --policy.oracle_depth="${ORACLE_DEPTH}" \
      --policy.oracle_n_heads="${ORACLE_N_HEADS}" \
      --policy.oracle_n_tokens="${ORACLE_N_TOKENS}" \
      --policy.oracle_r_dim="${ORACLE_R_DIM}" \
      --policy.oracle_free_bits="${ORACLE_FREE_BITS}" \
      --policy.oracle_kl_weight="${ORACLE_KL_WEIGHT}" \
      --policy.oracle_dropout_p="${ORACLE_DROPOUT_P}" \
      --policy.r_ablation_every="${R_ABLATION_EVERY}" \
      --policy.freeze_vsa_base="${FREEZE_VSA}" \
      --policy.freeze_vsa_vision="${FREEZE_VSA_VISION}" \
      --policy.boundary_mode="${BOUNDARY_MODE}" \
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
