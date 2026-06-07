#!/usr/bin/env bash
# Inputs:
#   roots      : ../../train_skills_config.yaml
#   eval knobs : ../fsq_eval_config.yaml
#   FSQ model  : {project_root}/FSQ_outputs/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch{N:04d}.pt)
# Outputs:
#   {FSQ_eval}/outputs/{run}/{epoch}/skill_latents.npz
#   {FSQ_eval}/outputs/{run}/{epoch}/fsq_eval.html
#
# Unified FSQ evaluation (encoder + decoder) → wandb scalars + one interactive HTML.
#SBATCH --job-name=FSQ_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Under Slurm the script runs from a spool copy, so ${BASH_SOURCE[0]} is unreliable.
# Anchor on SLURM_SUBMIT_DIR (the FSQ_eval dir the submit script cd'd into).
EVAL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # FSQ_eval
COMMON_SRC_DIR="${EVAL_DIR}/../src"                          # train_skills/src
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${EVAL_DIR}/../train_skills_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${EVAL_DIR}/fsq_eval_config.yaml}"

BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/fsq_eval_config.py" --config "${EVAL_CONFIG}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=""
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${PROJECT_ROOT}/../.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${PROJECT_ROOT}/../.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

FSQ_OUTPUTS_ROOT="$(dirname "${FSQ_OUTPUT_DIR}")"
MODEL_DIR="${FSQ_OUTPUTS_ROOT}/${FSQ_EVAL_RUN_NAME}"
DATASET_VIDEO_DIR="${DATASET_ROOT}/${TARGET_DATASET}"

if [ "${FSQ_EVAL_CHECKPOINT}" = "0" ]; then
  MODEL_PATH="${MODEL_DIR}/FSQ.pt"
  EPOCH_TAG="best"
  LATENTS_PATH="${MODEL_DIR}/skill_latents.npz"
else
  MODEL_PATH="${MODEL_DIR}/FSQ_epoch$(printf '%04d' "${FSQ_EVAL_CHECKPOINT}").pt"
  EPOCH_TAG="epoch$(printf '%04d' "${FSQ_EVAL_CHECKPOINT}")"
  LATENTS_PATH="${MODEL_DIR}/skill_latents_${EPOCH_TAG}.npz"
fi
# HTML stays under FSQ_eval/outputs; latents live next to the checkpoint in FSQ_outputs.
OUTPUT_DIR="${EVAL_DIR}/outputs/${FSQ_EVAL_RUN_NAME}/${EPOCH_TAG}"

echo "START FSQ EVAL"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "Model   : ${MODEL_PATH}"
echo "Skills  : ${SKILLSET_DIR}/skills"
echo "DINO 3rd: ${DINO_TOKENS_PATH}"
echo "DINO wri: ${DINO_TOKENS_WRIST_PATH}"
echo "Dataset : ${DATASET_VIDEO_DIR}"
echo "Latents : ${LATENTS_PATH}"
echo "Output  : ${OUTPUT_DIR}"
nvidia-smi || true

if [ ! -f "${MODEL_PATH}" ]; then
  echo "FSQ model not found: ${MODEL_PATH}" >&2
  echo "Check fsq_eval_run_name / checkpoint in ${EVAL_CONFIG}" >&2
  exit 1
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u examples/libero/fsq_eval.py \
    --model_path          "${MODEL_PATH}" \
    --skills_dir          "${SKILLSET_DIR}/skills" \
    --dino_features       "${DINO_TOKENS_PATH}" \
    --dino_features_wrist "${DINO_TOKENS_WRIST_PATH}" \
    --dataset_dir      "${DATASET_VIDEO_DIR}" \
    --image_key        "${FSQ_EVAL_IMAGE_KEY}" \
    --output_dir       "${OUTPUT_DIR}" \
    --latents_path     "${LATENTS_PATH}" \
    --n_action_steps   "${FSQ_EVAL_N_ACTION_STEPS}" \
    --max_plot_samples "${FSQ_EVAL_N_SAMPLES}" \
    --max_plot_entries "${FSQ_EVAL_MAX_ENTRIES}" \
    --thumb_size       "${FSQ_EVAL_THUMB_SIZE}" \
    --end_threshold    "${FSQ_EVAL_END_THRESHOLD}" \
    --wandb_project    "${FSQ_EVAL_WANDB_PROJECT}" \
    --wandb_run_name   "${FSQ_EVAL_RUN_NAME}_${EPOCH_TAG}"

echo "DONE → ${OUTPUT_DIR}/fsq_eval.html"
