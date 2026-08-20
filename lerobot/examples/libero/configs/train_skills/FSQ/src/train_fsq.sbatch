#!/usr/bin/env bash
# Inputs:
#   skillset     : exact folder selected in fsq_config.yaml and validated against
#                  its skillset_manifest.json before submission
#   Images       : selected raw frames only for image/both terminator input
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/outputs/FSQ/{fsq_run_name}
#   checkpoints  : {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt
#   best model   : optional FSQ.pt when fsq_save_best_model=true
#   skill tokens : {project_root}/outputs/FSQ/{fsq_run_name}/skill_latents.npz
#
# Train FSQ from prepared FSQ_inputs.
#SBATCH --job-name=FSQ_train
#SBATCH --gres=gpu:1
#SBATCH --requeue
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

# Config resolution is stdlib-only (with a built-in YAML fallback), so do not
# traverse the project .venv until the actual training process starts.
BOOTSTRAP_PYTHON=/usr/bin/python3

if [ -n "${TRAIN_DATA:-}" ]; then
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "${RESOLVED_SETTINGS}"

cd "${LEROBOT_ROOT}"
SHARED_VENV="$(readlink -f "${PROJECT_ROOT}/.venv")"
TRAIN_VENV="${SHARED_VENV}"
source "${SUBMIT_DIR}/src/fsq_node_local_venv.sh"
if [ -n "${FSQ_VENV_ARCHIVE:-}" ]; then
  if STAGED_VENV="$(fsq_stage_venv_on_node "${FSQ_VENV_ARCHIVE}" "${SHARED_VENV}")"; then
    TRAIN_VENV="${STAGED_VENV}"
  fi
fi
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
export VIRTUAL_ENV="${TRAIN_VENV}"
export PATH="${TRAIN_VENV}/bin:${PATH}"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""
# Reduce CUDA fragmentation: activation tensor sizes vary per batch (max skill length).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START FSQ TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Inputs     : ${FSQ_INPUTS_DIR}"
echo "Output     : ${FSQ_OUTPUT_DIR}"
echo "HF cache   : ${HF_HOME}"
echo "Python     : ${TRAIN_PYTHON}"
echo "Grad ckpt  : ${FSQ_GRADIENT_CHECKPOINTING}"
nvidia-smi

# Fail over to another GPU instead of silently running on CPU. The Python
# entrypoint reuses its first torch import, avoiding a duplicate cold import.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
prepare_inline_cuda_guard

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi
if { [ "${FSQ_DECODER_TERMINATOR_PROGRESS}" = "true" ] || [ "${FSQ_DECODER_TERMINATOR_TERMINATION}" = "true" ]; } \
  && [ "${FSQ_TERMINATOR_INPUT_SPACE}" != "state" ] \
  && [ ! -d "${RAW_DATASET_DIR}/videos" ]; then
  echo "Raw dataset videos not found: ${RAW_DATASET_DIR}/videos" >&2
  exit 1
fi
if [ "${FSQ_FREEZE_VISION_ENCODER}" = "true" ]; then
  VISION_FREEZE_ARG=(--freeze_vision_encoder)
else
  VISION_FREEZE_ARG=(--no-freeze_vision_encoder)
fi

read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"

mkdir -p "${FSQ_OUTPUT_DIR}"

# Provenance so the eval can auto-evaluate this FSQ on the SAME skillset/cameras it was trained on,
# without the user re-specifying the DP (fsq_run_name doesn't encode the DP).
cat > "${FSQ_OUTPUT_DIR}/fsq_meta.json" <<JSON
{
  "fsq_dataset_root": "${FSQ_DATASET_ROOT_NAME}",
  "target_dataset": "${TARGET_DATASET}",
  "fsq_inputs_name": "${FSQ_INPUTS_NAME}",
  "skillset_seg_name": "${SKILLSET_SEG_NAME}",
  "skillset_name": "${SKILLSET_NAME}",
  "dp_run_name": "${DP_POLICY}",
  "dp_checkpoint": "${DP_CHECKPOINT}",
  "skillset_mode": "${SKILLSET_MODE}",
  "skillset_min_skills": ${SKILLSET_MIN_SKILLS},
  "skillset_min_skill_len": ${SKILLSET_MIN_SKILL_LEN},
  "skillset_boundary_threshold_mode": "${SKILLSET_BOUNDARY_THRESHOLD_MODE}",
  "skillset_boundary_threshold_scale": ${SKILLSET_BOUNDARY_THRESHOLD_SCALE},
  "skillset_global_threshold_source": "${SKILLSET_GLOBAL_THRESHOLD_SOURCE}",
  "skillset_output_suffix": "${SKILLSET_OUTPUT_SUFFIX}",
  "decoder_reconstructor": ${FSQ_DECODER_RECONSTRUCTOR},
  "decoder_terminator_progress": ${FSQ_DECODER_TERMINATOR_PROGRESS},
  "decoder_terminator_termination": ${FSQ_DECODER_TERMINATOR_TERMINATION},
  "terminator_input_space": "${FSQ_TERMINATOR_INPUT_SPACE}",
  "terminator_arch": "${FSQ_TERMINATOR_MODEL}",
  "terminator_default_arch": "${FSQ_TERMINATOR_DEFAULT_ARCH}",
  "terminator_layers": ${FSQ_TERMINATOR_LAYERS},
  "terminator_heads": ${FSQ_TERMINATOR_HEADS},
  "skill_cond_mode": "${FSQ_SKILL_COND_MODE}",
  "gradient_checkpointing": ${FSQ_GRADIENT_CHECKPOINTING},
  "lr_schedule": "${FSQ_LR_SCHEDULE}",
  "num_workers": ${FSQ_NUM_WORKERS},
  "val_num_workers": ${FSQ_VAL_NUM_WORKERS},
  "val_every": ${FSQ_VAL_EVERY},
  "save_best_model": ${FSQ_SAVE_BEST_MODEL},
  "vision_backbone": "${FSQ_VISION_BACKBONE}",
  "freeze_vision_encoder": ${FSQ_FREEZE_VISION_ENCODER},
  "encoder_input_mode": "${FSQ_ENCODER_INPUT_MODE}",
  "encoder_arch": "${FSQ_ENCODER_ARCH}",
  "encoder_length_token": ${FSQ_ENCODER_LENGTH_TOKEN},
  "fsq_entropy": ${FSQ_ENTROPY},
  "reconstructor_start_state": ${FSQ_RECONSTRUCTOR_START_STATE},
  "reconstructor_arch": "${FSQ_RECONSTRUCTOR_ARCH}"
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

GRADIENT_CKPT_ARG=(--no-gradient_checkpointing)
if [ "${FSQ_GRADIENT_CHECKPOINTING:-false}" = "true" ]; then
  GRADIENT_CKPT_ARG=(--gradient_checkpointing)
fi

SAVE_BEST_ARG=(--no-save_best_model)
if [ "${FSQ_SAVE_BEST_MODEL:-false}" = "true" ]; then
  SAVE_BEST_ARG=(--save_best_model)
fi

RECON_START_STATE_ARG=(--reconstructor_start_state)
if [ "${FSQ_RECONSTRUCTOR_START_STATE:-true}" = "false" ]; then
  RECON_START_STATE_ARG=(--no-reconstructor_start_state)
fi

FSQ_ENTROPY_ARG=(--no-fsq_entropy)
if [ "${FSQ_ENTROPY:-false}" = "true" ]; then
  FSQ_ENTROPY_ARG=(--fsq_entropy)
fi

DECODER_RECONSTRUCTOR_ARG=(--no-decoder_reconstructor)
if [ "${FSQ_DECODER_RECONSTRUCTOR}" = "true" ]; then
  DECODER_RECONSTRUCTOR_ARG=(--decoder_reconstructor)
fi

DECODER_PROGRESS_ARG=(--no-decoder_terminator_progress)
if [ "${FSQ_DECODER_TERMINATOR_PROGRESS}" = "true" ]; then
  DECODER_PROGRESS_ARG=(--decoder_terminator_progress)
fi

DECODER_TERMINATION_ARG=(--no-decoder_terminator_termination)
if [ "${FSQ_DECODER_TERMINATOR_TERMINATION}" = "true" ]; then
  DECODER_TERMINATION_ARG=(--decoder_terminator_termination)
fi

# best-val SELECTION override: pass each val_select weight only if set in config (empty → omit → the
# trainer falls back to selecting FSQ.pt on action+progress+end with their weights).
VAL_SELECT_ARGS=()
if [ -n "${FSQ_VAL_SELECT_ACTION_WEIGHT:-}" ];   then VAL_SELECT_ARGS+=(--val_select_action_weight   "${FSQ_VAL_SELECT_ACTION_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_PROGRESS_WEIGHT:-}" ]; then VAL_SELECT_ARGS+=(--val_select_progress_weight "${FSQ_VAL_SELECT_PROGRESS_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_END_WEIGHT:-}" ];      then VAL_SELECT_ARGS+=(--val_select_end_weight      "${FSQ_VAL_SELECT_END_WEIGHT}"); fi

TRAIN_STATUS=0
PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  "${TRAIN_PYTHON}" -u examples/libero/train_FSQ.py \
    --skills_dir              "${SKILLSET_DIR}/skills" \
    --raw_dataset_dir         "${RAW_DATASET_DIR}" \
    --output_dir              "${FSQ_OUTPUT_DIR}" \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --chunk_size              "${FSQ_CHUNK_SIZE}" \
    --samples_per_skill       "${FSQ_SAMPLES_PER_SKILL}" \
    --pi_base                 "${PI_BASE_MODEL_PATH}" \
    --skill_cond_mode         "${FSQ_SKILL_COND_MODE}" \
    --hidden_dim              "${FSQ_HIDDEN_DIM}" \
    --num_layers              "${FSQ_NUM_LAYERS}" \
    --n_control               "${FSQ_N_CONTROL}" \
    --encoder_input_mode      "${FSQ_ENCODER_INPUT_MODE}" \
    --encoder_arch            "${FSQ_ENCODER_ARCH}" \
    "${FSQ_ENTROPY_ARG[@]}" \
    "${RECON_START_STATE_ARG[@]}" \
    --reconstructor_arch      "${FSQ_RECONSTRUCTOR_ARCH}" \
    --entropy_conf_weight     "${FSQ_ENTROPY_CONF_WEIGHT}" \
    --entropy_div_weight      "${FSQ_ENTROPY_DIV_WEIGHT}" \
    --entropy_inv_temperature "${FSQ_ENTROPY_INV_TEMPERATURE}" \
    "${DECODER_RECONSTRUCTOR_ARG[@]}" \
    "${DECODER_PROGRESS_ARG[@]}" \
    "${DECODER_TERMINATION_ARG[@]}" \
    --terminator_input_space  "${FSQ_TERMINATOR_INPUT_SPACE}" \
    --terminator_model        "${FSQ_TERMINATOR_MODEL}" \
    --visual_terminator_arch  "${FSQ_TERMINATOR_DEFAULT_ARCH}" \
    --image_encoder_layers    "${FSQ_TERMINATOR_LAYERS}" \
    --image_encoder_heads     "${FSQ_TERMINATOR_HEADS}" \
    --vision_backbone         "${FSQ_VISION_BACKBONE}" \
    "${VISION_FREEZE_ARG[@]}" \
    --dino_model_path         "${FSQ_DINO_MODEL_PATH}" \
    --dino_image_size         "${FSQ_DINO_IMAGE_SIZE}" \
    --siglip_image_size       "${FSQ_SIGLIP_IMAGE_SIZE}" \
    --cond_encoder_variant    "${FSQ_COND_ENCODER_VARIANT}" \
    --epochs                  "${FSQ_NUM_EPOCHS}" \
    --encoder_lr              "${FSQ_ENCODER_LR}" \
    --terminator_lr           "${FSQ_TERMINATOR_LR}" \
    --reconstructor_lr        "${FSQ_RECONSTRUCTOR_LR}" \
    --lr_schedule             "${FSQ_LR_SCHEDULE}" \
    --batch_size              "${FSQ_BATCH_SIZE}" \
    --num_workers             "${FSQ_NUM_WORKERS}" \
    --val_num_workers         "${FSQ_VAL_NUM_WORKERS}" \
    --val_every               "${FSQ_VAL_EVERY}" \
    "${SAVE_BEST_ARG[@]}" \
    "${GRADIENT_CKPT_ARG[@]}" \
    --action_loss_weight      "${FSQ_ACTION_LOSS_WEIGHT}" \
    --progress_loss_weight    "${FSQ_PROGRESS_LOSS_WEIGHT}" \
    --end_loss_weight         "${FSQ_END_LOSS_WEIGHT}" \
    "${VAL_SELECT_ARGS[@]}" \
    --end_target_sigma        "${FSQ_END_TARGET_SIGMA}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}" || TRAIN_STATUS=$?

handle_inline_cuda_guard_exit "${TRAIN_STATUS}"
if [ "${TRAIN_STATUS}" -ne 0 ]; then
  exit "${TRAIN_STATUS}"
fi

echo "DONE ${FSQ_RUN_NAME}"
