#!/usr/bin/env bash
# Inputs:
#   FSQ inputs   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
#   Images       : selected raw third+wrist frames, decoded live per batch
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ-new run  : {project_root}/outputs/FSQ_new/{fsq_run_name}
#   FSQ model    : {project_root}/outputs/FSQ_new/{fsq_run_name}/FSQ.pt
#   checkpoints  : {project_root}/outputs/FSQ_new/{fsq_run_name}/FSQ_epoch*.pt
#
# Train FSQ from prepared FSQ_inputs.
#SBATCH --job-name=FSQ_train
#SBATCH --gres=gpu:1
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
FSQ_NEW_SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/fsq_new_config.yaml}

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "$("${BOOTSTRAP_PYTHON}" "${FSQ_NEW_SRC_DIR}/resolve_fsq_new_config.py" --config "${CONFIG_PATH}")"
export FSQ_OUTPUT_DIR="${OUTPUTS_ROOT}/FSQ_new/${FSQ_RUN_NAME}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""
# Reduce CUDA fragmentation: activation tensor sizes vary per batch (max skill length).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START FSQ TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Inputs     : ${FSQ_INPUTS_DIR}"
echo "Output     : ${FSQ_OUTPUT_DIR}"
echo "HF cache   : ${HF_HOME}"
nvidia-smi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi
if [ ! -d "${RAW_DATASET_DIR}/videos" ]; then
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
  "dp_run_name": "${DP_POLICY}",
  "dp_checkpoint": "${DP_CHECKPOINT}",
  "skillset_mode": "${SKILLSET_MODE}",
  "skillset_min_skills": ${SKILLSET_MIN_SKILLS},
  "skillset_boundary_threshold_mode": "${SKILLSET_BOUNDARY_THRESHOLD_MODE}",
  "skillset_global_threshold_source": "${SKILLSET_GLOBAL_THRESHOLD_SOURCE}",
  "skillset_output_suffix": "${SKILLSET_OUTPUT_SUFFIX}",
  "weighted_loss": ${FSQ_WEIGHTED_LOSS},
  "weighted_loss_end_weight": ${FSQ_WEIGHTED_LOSS_END_WEIGHT},
  "terminator_arch": "${FSQ_TERMINATOR_ARCH}",
  "terminator_layers": ${FSQ_TERMINATOR_LAYERS},
  "terminator_heads": ${FSQ_TERMINATOR_HEADS},
  "state_cond_mode": "${FSQ_STATE_COND_MODE}",
  "vision_backbone": "${FSQ_VISION_BACKBONE}",
  "freeze_vision_encoder": ${FSQ_FREEZE_VISION_ENCODER},
  "encoder_input_mode": "${FSQ_ENCODER_INPUT_MODE}"
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

# weighted_loss probe (tyro bool flag): end-weight the sampled flow loss.
WEIGHTED_ARG=(--no-weighted_loss)
if [ "${FSQ_WEIGHTED_LOSS:-false}" = "true" ]; then
  WEIGHTED_ARG=(--weighted_loss)
fi

WRONG_GOAL_ARG=(--no-wrong_goal_enabled)
if [ "${FSQ_NEW_WRONG_GOAL_ENABLED}" = "true" ]; then
  WRONG_GOAL_ARG=(--wrong_goal_enabled)
fi

# best-val SELECTION override: pass each val_select weight only if set in config (empty → omit → the
# trainer falls back to selecting FSQ.pt on action+progress+end with their weights).
VAL_SELECT_ARGS=()
if [ -n "${FSQ_VAL_SELECT_ACTION_WEIGHT:-}" ];   then VAL_SELECT_ARGS+=(--val_select_action_weight   "${FSQ_VAL_SELECT_ACTION_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_PROGRESS_WEIGHT:-}" ]; then VAL_SELECT_ARGS+=(--val_select_progress_weight "${FSQ_VAL_SELECT_PROGRESS_WEIGHT}"); fi
if [ -n "${FSQ_VAL_SELECT_END_WEIGHT:-}" ];      then VAL_SELECT_ARGS+=(--val_select_end_weight      "${FSQ_VAL_SELECT_END_WEIGHT}"); fi

PYTHONPATH="${FSQ_NEW_SRC_DIR}:${PROJECT_ROOT}/lerobot/src" \
  python -u "${FSQ_NEW_SRC_DIR}/train_FSQ_new.py" \
    --skills_dir              "${SKILLSET_DIR}/skills" \
    --raw_dataset_dir         "${RAW_DATASET_DIR}" \
    --output_dir              "${FSQ_OUTPUT_DIR}" \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --chunk_size              "${FSQ_CHUNK_SIZE}" \
    --samples_per_skill       "${FSQ_SAMPLES_PER_SKILL}" \
    --action_expert_variant   "${FSQ_ACTION_EXPERT_VARIANT}" \
    --state_cond_mode         "${FSQ_STATE_COND_MODE}" \
    --pi_base                 "${PI_BASE_MODEL_PATH}" \
    --expert_dtype            "${FSQ_EXPERT_DTYPE}" \
    --context_queries         "${FSQ_NEW_CONTEXT_QUERIES}" \
    --context_layers          "${FSQ_NEW_CONTEXT_LAYERS}" \
    --context_heads           "${FSQ_NEW_CONTEXT_HEADS}" \
    --context_dropout         "${FSQ_NEW_CONTEXT_DROPOUT}" \
    --a_skill_scale           "${FSQ_NEW_A_SKILL_SCALE}" \
    --b_skill_scale           "${FSQ_NEW_B_SKILL_SCALE}" \
    --c_skill_scale           "${FSQ_NEW_C_SKILL_SCALE}" \
    --b_image_scale           "${FSQ_NEW_B_IMAGE_SCALE}" \
    --c_image_scale           "${FSQ_NEW_C_IMAGE_SCALE}" \
    --c_goal_scale            "${FSQ_NEW_C_GOAL_SCALE}" \
    --hidden_dim              "${FSQ_HIDDEN_DIM}" \
    --num_layers              "${FSQ_NUM_LAYERS}" \
    --n_control               "${FSQ_N_CONTROL}" \
    --encoder_input_mode      "${FSQ_ENCODER_INPUT_MODE}" \
    --terminator_arch         "${FSQ_TERMINATOR_ARCH}" \
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
    --expert_lr               "${FSQ_EXPERT_LR}" \
    --context_lr              "${FSQ_NEW_CONTEXT_LR}" \
    --batch_size              "${FSQ_BATCH_SIZE}" \
    --num_workers             "${FSQ_NUM_WORKERS}" \
    --action_loss_weight      "${FSQ_ACTION_LOSS_WEIGHT}" \
    --progress_loss_weight    "${FSQ_PROGRESS_LOSS_WEIGHT}" \
    --end_loss_weight         "${FSQ_END_LOSS_WEIGHT}" \
    --regime_a_weight         "${FSQ_NEW_A_WEIGHT}" \
    --regime_b_weight         "${FSQ_NEW_B_WEIGHT}" \
    --regime_c_weight         "${FSQ_NEW_C_WEIGHT}" \
    --ranking_weight          "${FSQ_NEW_RANKING_WEIGHT}" \
    --ranking_relative_margin "${FSQ_NEW_RANKING_MARGIN}" \
    "${WRONG_GOAL_ARG[@]}" \
    --wrong_goal_weight       "${FSQ_NEW_WRONG_GOAL_WEIGHT}" \
    --wrong_goal_relative_margin "${FSQ_NEW_WRONG_GOAL_MARGIN}" \
    "${VAL_SELECT_ARGS[@]}" \
    --end_target_sigma        "${FSQ_END_TARGET_SIGMA}" \
    "${WEIGHTED_ARG[@]}" \
    --weighted_loss_end_weight "${FSQ_WEIGHTED_LOSS_END_WEIGHT}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE ${FSQ_RUN_NAME}"
