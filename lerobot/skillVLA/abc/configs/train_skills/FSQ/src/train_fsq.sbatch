#!/usr/bin/env bash
#SBATCH --job-name=ABC_FSQ
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs_train/%x_%j.out
#SBATCH --error=logs_train/%x_%j.err
set -euo pipefail
mkdir -p logs_train

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_FSQ_CONFIG:-${SUBMIT_DIR}/fsq_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../src/train_skills_config.py" \
  --config "${CONFIG_PATH}" --stage fsq --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/skillVLA/abc/configs/src/runtime_env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Preparing/reusing ${ABC_SOURCE_TASKS_LABEL} native source statistics (${FSQ_ACTION_SPACE})..."
FSQ_STATS_PATH="$(
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python "${SUBMIT_DIR}/src/prepare_native_fsq_stats.py" \
      --dataset-dir "${RAW_DATASET_DIR}" \
      --tasks-json "${ABC_SOURCE_TASKS_JSON}"
)"
[ -f "${FSQ_STATS_PATH}" ] || { echo "FSQ stats not found: ${FSQ_STATS_PATH}" >&2; exit 1; }

echo "START ABC FSQ"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "Skills    : ${SKILLSET_DIR}/skills"
echo "Stats     : ${FSQ_STATS_PATH}"
echo "Action    : ${FSQ_ACTION_SPACE} ${FSQ_ACTION_DIM}D"
echo "Grippers  : ${FSQ_ACTION_GRIPPER_INDICES_STR} (weight=${FSQ_ACTION_GRIPPER_WEIGHT})"
echo "Output    : ${FSQ_OUTPUT_DIR}"
echo "schedule  : epochs=${FSQ_NUM_EPOCHS}"
nvidia-smi || true

read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
read -r -a GRIPPER_INDICES <<< "${FSQ_ACTION_GRIPPER_INDICES_STR}"
mkdir -p "${FSQ_OUTPUT_DIR}"
cat > "${FSQ_OUTPUT_DIR}/fsq_meta.json" <<JSON
{
  "target_dataset": "${TARGET_DATASET}",
  "source_tasks": ${ABC_SOURCE_TASKS_JSON},
  "skillset_seg_name": "${SKILLSET_SEG_NAME}",
  "skillset_manifest_path": "${SKILLSET_MANIFEST_PATH}",
  "action_space": "${FSQ_ACTION_SPACE}",
  "encoder_mode": "${FSQ_ENCODER_MODE}",
  "action_dim": ${FSQ_ACTION_DIM},
  "action_gripper_indices": [${FSQ_ACTION_GRIPPER_INDICES_STR// /, }],
  "action_gripper_weight": ${FSQ_ACTION_GRIPPER_WEIGHT},
  "stats_path": "${FSQ_STATS_PATH}",
  "action_stats_path": "${FSQ_OUTPUT_DIR}/action_stats.npz",
  "autoencoder_mode": "${FSQ_AUTOENCODER_MODE}",
  "quantizer": "${FSQ_QUANTIZER}",
  "fsq_levels": [${FSQ_LEVELS_STR// /, }],
  "boundary_aug_pmax_percent": ${FSQ_BOUNDARY_AUG_PMAX_PERCENT},
  "boundary_aug_preserve_gripper": ${FSQ_BOUNDARY_AUG_PRESERVE_GRIPPER},
  "boundary_aug_directional": {
    "early_start": ${FSQ_BOUNDARY_AUG_EARLY_START_PMAX},
    "late_start": ${FSQ_BOUNDARY_AUG_LATE_START_PMAX},
    "early_end": ${FSQ_BOUNDARY_AUG_EARLY_END_PMAX},
    "late_end": ${FSQ_BOUNDARY_AUG_LATE_END_PMAX}
  }
}
JSON

RESUME_ARG=()
LATEST_CHECKPOINT="$(find "${FSQ_OUTPUT_DIR}" -maxdepth 1 -type f -name 'FSQ_epoch*.pt' -printf '%f\n' \
  | sort -V | tail -1)"
if [ -n "${LATEST_CHECKPOINT}" ]; then
  RESUME_ARG=(--resume_from "${FSQ_OUTPUT_DIR}/${LATEST_CHECKPOINT}")
  echo "Resume    : ${RESUME_ARG[1]}"
fi

INIT_CALIBRATION_ARG=(--no-init_calibration)
[ "${FSQ_INIT_CALIBRATION}" = true ] && INIT_CALIBRATION_ARG=(--init_calibration)
PAIR_WARMUP_ARG=(--no-pair_warmup)
[ "${FSQ_PAIR_WARMUP}" = true ] && PAIR_WARMUP_ARG=(--pair_warmup)
ROUTE_LOSS_ARG=(--no-route_loss)
[ "${FSQ_ROUTE_LOSS}" = true ] && ROUTE_LOSS_ARG=(--route_loss)
BOUNDARY_PRESERVE_ARG=(--no-boundary_aug_preserve_gripper)
[ "${FSQ_BOUNDARY_AUG_PRESERVE_GRIPPER}" = true ] && BOUNDARY_PRESERVE_ARG=(--boundary_aug_preserve_gripper)

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u skillVLA/libero/configs/train_skills/FSQ/src/train_FSQ.py \
    --skills_dir "${SKILLSET_DIR}/skills" \
    --raw_dataset_dir "${RAW_DATASET_DIR}" \
    --stats_path "${FSQ_STATS_PATH}" \
    --action_space "${FSQ_ACTION_SPACE}" \
    --action_encoder_mode "${FSQ_ENCODER_MODE}" \
    --output_dir "${FSQ_OUTPUT_DIR}" \
    --quantizer "${FSQ_QUANTIZER}" \
    --bsq_code_dim "${BSQ_CODE_DIM}" \
    --fsq_levels "${FSQ_LEVELS[@]}" \
    --hidden_dim "${FSQ_HIDDEN_DIM}" \
    --num_layers "${FSQ_NUM_LAYERS}" \
    --n_control "${FSQ_N_CONTROL}" \
    --autoencoder_mode "${FSQ_AUTOENCODER_MODE}" \
    --action_gripper_weight "${FSQ_ACTION_GRIPPER_WEIGHT}" \
    --action_gripper_indices "${GRIPPER_INDICES[@]}" \
    --start_state_conditioning "${FSQ_START_STATE_CONDITIONING}" \
    "${INIT_CALIBRATION_ARG[@]}" \
    --init_calibration_gain "${FSQ_INIT_CALIBRATION_GAIN}" \
    --init_calibration_samples "${FSQ_INIT_CALIBRATION_SAMPLES}" \
    --pair_loss "${FSQ_PAIR_LOSS}" \
    --pair_weight "${FSQ_PAIR_WEIGHT}" \
    --pair_inv_temperature "${FSQ_PAIR_INV_TEMPERATURE}" \
    "${ROUTE_LOSS_ARG[@]}" \
    "${PAIR_WARMUP_ARG[@]}" \
    --pair_warmup_epochs "${FSQ_PAIR_WARMUP_EPOCHS}" \
    --pair_ramp_epochs "${FSQ_PAIR_RAMP_EPOCHS}" \
    --boundary_aug_pmax "${FSQ_BOUNDARY_AUG_PMAX}" \
    --boundary_aug_pmax_percent "${FSQ_BOUNDARY_AUG_PMAX_PERCENT}" \
    "${BOUNDARY_PRESERVE_ARG[@]}" \
    --boundary_aug_early_start_pmax "${FSQ_BOUNDARY_AUG_EARLY_START_PMAX}" \
    --boundary_aug_late_start_pmax "${FSQ_BOUNDARY_AUG_LATE_START_PMAX}" \
    --boundary_aug_early_end_pmax "${FSQ_BOUNDARY_AUG_EARLY_END_PMAX}" \
    --boundary_aug_late_end_pmax "${FSQ_BOUNDARY_AUG_LATE_END_PMAX}" \
    --boundary_aug_distribution "${FSQ_BOUNDARY_AUG_DISTRIBUTION}" \
    --decoder_reconstructor \
    --no-decoder_terminator_progress \
    --no-decoder_terminator_termination \
    --epochs "${FSQ_NUM_EPOCHS}" \
    --lr "${FSQ_LR}" \
    --lr_schedule "${FSQ_LR_SCHEDULE}" \
    --batch_size "${FSQ_BATCH_SIZE}" \
    --num_workers "${FSQ_NUM_WORKERS}" \
    --val_num_workers "${FSQ_VAL_NUM_WORKERS}" \
    --val_every "${FSQ_VAL_EVERY}" \
    --no-save_best_model \
    --no-gradient_checkpointing \
    --samples_per_skill "${FSQ_SAMPLES_PER_SKILL}" \
    --action_loss_weight "${FSQ_ACTION_LOSS_WEIGHT}" \
    --checkpoint_every "${FSQ_CHECKPOINT_EVERY}" \
    --wandb_project "${FSQ_WANDB_PROJECT}" \
    --wandb_run_name "${FSQ_RUN_NAME}" \
    "${RESUME_ARG[@]}"

echo "DONE -> ${FSQ_OUTPUT_DIR}"
