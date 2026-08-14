#!/usr/bin/env bash
# Inputs:
#   skillset     : exact folder selected in fsq_original_config.yaml and validated
#                  against its skillset_manifest.json before submission
# Outputs (same outputs/FSQ root as v3; run identity comes from fsq_exp):
#   run          : {project_root}/outputs/FSQ/{fsq_run_name}
#   checkpoints  : {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt
#   best model   : optional FSQ.pt when fsq_save_best_model=true
#
# Train the one-shot (original-style) FSQ trajectory autoencoder.
#SBATCH --job-name=FSQ_orig_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train/%x_%j.out
#SBATCH --error=logs_train/%x_%j.err
set -euo pipefail
mkdir -p logs_train

# Under Slurm the script runs from a spool copy, so ${BASH_SOURCE[0]} is unreliable.
# Anchor on SLURM_SUBMIT_DIR (the FSQ_original dir the submit script cd'd into);
# fall back to this file's parent for direct execution.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FSQ_ORIG_SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/fsq_original_config.yaml}

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${FSQ_ORIG_SRC_DIR}/fsq_original_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${FSQ_ORIG_SRC_DIR}/fsq_original_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "${RESOLVED_SETTINGS}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START FSQ-ORIGINAL TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
echo "Skillset   : ${SKILLSET_DIR}/skills"
echo "Output     : ${FSQ_OUTPUT_DIR}"
nvidia-smi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"

mkdir -p "${FSQ_OUTPUT_DIR}"

# Provenance so downstream tooling can trace WHICH skillset/DP this run used
# (fsq_run_name doesn't encode the DP).
cat > "${FSQ_OUTPUT_DIR}/fsq_original_meta.json" <<JSON
{
  "variant": "fsq_original_oneshot",
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
  "encoder_input_mode": "${FSQ_ENCODER_INPUT_MODE}",
  "encoder_arch": "${FSQ_ORIG_ENCODER_ARCH}",
  "encoder_length_token": ${FSQ_ORIG_ENCODER_LENGTH_TOKEN},
  "fsq_entropy": ${FSQ_ORIG_ENTROPY},
  "bsq_entropy_joint": ${BSQ_ENTROPY_JOINT},
  "decoder_layers": ${FSQ_ORIG_DECODER_LAYERS},
  "decoder_arch": "${FSQ_ORIG_DECODER_ARCH}",
  "reconstruct_length": ${FSQ_ORIG_RECONSTRUCT_LENGTH},
  "lr_schedule": "${FSQ_LR_SCHEDULE}",
  "val_every": ${FSQ_VAL_EVERY},
  "save_best_model": ${FSQ_SAVE_BEST_MODEL}
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

SAVE_BEST_ARG=(--no-save_best_model)
if [ "${FSQ_SAVE_BEST_MODEL:-false}" = "true" ]; then
  SAVE_BEST_ARG=(--save_best_model)
fi

RECON_LENGTH_ARG=(--reconstruct_length)
if [ "${FSQ_ORIG_RECONSTRUCT_LENGTH:-true}" = "false" ]; then
  RECON_LENGTH_ARG=(--no-reconstruct_length)
fi

ENC_LENGTH_TOKEN_ARG=(--encoder_length_token)
if [ "${FSQ_ORIG_ENCODER_LENGTH_TOKEN:-true}" = "false" ]; then
  ENC_LENGTH_TOKEN_ARG=(--no-encoder_length_token)
fi

FSQ_ENTROPY_ARG=(--no-fsq_entropy)
if [ "${FSQ_ORIG_ENTROPY:-false}" = "true" ]; then
  FSQ_ENTROPY_ARG=(--fsq_entropy)
fi
ENTROPY_JOINT_ARG=(--bsq_entropy_joint)
if [ "${BSQ_ENTROPY_JOINT:-true}" = "false" ]; then
  ENTROPY_JOINT_ARG=(--no-bsq_entropy_joint)
fi

# best-val SELECTION override: pass each val_select weight only if set in config
# (empty → omit → selection follows the ctrl/length loss weights).
VAL_SELECT_ARGS=()
if [ -n "${FSQ_ORIG_VAL_SELECT_CTRL_WEIGHT:-}" ];   then VAL_SELECT_ARGS+=(--val_select_ctrl_weight   "${FSQ_ORIG_VAL_SELECT_CTRL_WEIGHT}"); fi
if [ -n "${FSQ_ORIG_VAL_SELECT_LENGTH_WEIGHT:-}" ]; then VAL_SELECT_ARGS+=(--val_select_length_weight "${FSQ_ORIG_VAL_SELECT_LENGTH_WEIGHT}"); fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/train_FSQ_original.py \
    --skills_dir              "${SKILLSET_DIR}/skills" \
    --output_dir              "${FSQ_OUTPUT_DIR}" \
    --fsq_levels              "${FSQ_LEVELS[@]}" \
    --hidden_dim              "${FSQ_HIDDEN_DIM}" \
    --num_layers              "${FSQ_NUM_LAYERS}" \
    --n_control               "${FSQ_N_CONTROL}" \
    --encoder_input_mode      "${FSQ_ENCODER_INPUT_MODE}" \
    --encoder_arch            "${FSQ_ORIG_ENCODER_ARCH}" \
    "${ENC_LENGTH_TOKEN_ARG[@]}" \
    --decoder_layers          "${FSQ_ORIG_DECODER_LAYERS}" \
    --decoder_arch            "${FSQ_ORIG_DECODER_ARCH}" \
    "${FSQ_ENTROPY_ARG[@]}" \
    "${ENTROPY_JOINT_ARG[@]}" \
    --bsq_inv_temperature     "${BSQ_INV_TEMPERATURE}" \
    --bsq_entropy_conf_weight "${BSQ_ENTROPY_CONF_WEIGHT}" \
    --bsq_entropy_div_weight  "${BSQ_ENTROPY_DIV_WEIGHT}" \
    "${RECON_LENGTH_ARG[@]}" \
    --ctrl_loss_weight        "${FSQ_ORIG_CTRL_LOSS_WEIGHT}" \
    --length_loss_weight      "${FSQ_ORIG_LENGTH_LOSS_WEIGHT}" \
    --action_loss_weight      "${FSQ_ORIG_ACTION_LOSS_WEIGHT}" \
    --term_loss_weight        "${FSQ_ORIG_TERM_LOSS_WEIGHT}" \
    --term_pos_weight         "${FSQ_ORIG_TERM_POS_WEIGHT}" \
    --term_sigma              "${FSQ_ORIG_TERM_SIGMA}" \
    --epochs                  "${FSQ_NUM_EPOCHS}" \
    --encoder_lr              "${FSQ_ENCODER_LR}" \
    --decoder_lr              "${FSQ_ORIG_DECODER_LR}" \
    --lr_schedule             "${FSQ_LR_SCHEDULE}" \
    --batch_size              "${FSQ_BATCH_SIZE}" \
    --num_workers             "${FSQ_NUM_WORKERS}" \
    --val_every               "${FSQ_VAL_EVERY}" \
    "${SAVE_BEST_ARG[@]}" \
    "${VAL_SELECT_ARGS[@]}" \
    --checkpoint_every        "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project           "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name          "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE ${FSQ_RUN_NAME}"
