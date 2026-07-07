#!/usr/bin/env bash
# pi05 FT training.
#
# Inputs:
#   dataset : {project_root}/{ft_dataset_root}/{ft_dataset}
#   base    : {project_root}/{pi05_outputs_root}/{ft_pretrained_run_name}/checkpoints/{ft_pretrained_checkpoint}/pretrained_model
# Output:
#   model   : {project_root}/{pi05_ft_outputs_root}/FT_{ft_pretrained_checkpoint}PT_{ft_dataset}_pi05_batch{ft_batch_size}_{ft_exp}
# Config:
#   ./pi05_config.yaml
#SBATCH --job-name=pi05_FT
#SBATCH --partition=gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"             # .../train_pi05/pi05
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"          # .../train_pi05
CONFIG_PATH="${TRAIN_PI05_CONFIG:-${SUBMIT_DIR}/pi05_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START pi05 FT"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${FT_DATASET_DIR}"
echo "base     : ${FT_PRETRAINED_MODEL_PATH}"
echo "output   : ${FT_OUTPUT_DIR}"
echo "freeze   : vision=${FT_FREEZE_VISION_ENCODER} lang=${FT_FREEZE_LANGUAGE_MODEL}"
nvidia-smi || true

if [ ! -d "${FT_DATASET_DIR}" ]; then
  echo "Dataset not found: ${FT_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -d "${FT_PRETRAINED_MODEL_PATH}" ]; then
  echo "PT checkpoint not found: ${FT_PRETRAINED_MODEL_PATH}" >&2
  exit 1
fi

CONFIG_JSON="${FT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found. Resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
fi

# PT-forgetting probe (skillVLA FT와 동일 인프라): 고정 PT-dataset 배치를 PROBE_EVERY 스텝마다
# forward-only 재측정 → wandb probe/* + probe_forget/*. Empty root = probe off.
PROBE_ARGS=()
if [ -n "${FT_PROBE_DATASET_ROOT}" ]; then
  PROBE_ARGS=(
    --probe_dataset_root="${FT_PROBE_DATASET_ROOT}"
    --probe_dataset_repo_id="${FT_PROBE_DATASET_REPO_ID}"
    --probe_every="${FT_PROBE_EVERY}"
    --probe_batches="${FT_PROBE_BATCHES}"
    --probe_seed="${FT_PROBE_SEED}"
  )
fi

PYTORCH_ALLOC_CONF=expandable_segments:True "${TRAIN_BIN}" \
  --dataset.repo_id="lerobot/${FT_DATASET}" \
  --dataset.root="${FT_DATASET_DIR}" \
  --policy.type=pi05 \
  --output_dir="${FT_OUTPUT_DIR}" \
  --job_name="${FT_RUN_NAME}" \
  --policy.pretrained_path="${FT_PRETRAINED_MODEL_PATH}" \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.project="${FT_WANDB_PROJECT}" \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder="${FT_FREEZE_VISION_ENCODER}" \
  --policy.freeze_language_model="${FT_FREEZE_LANGUAGE_MODEL}" \
  --policy.push_to_hub=false \
  --save_freq="${FT_SAVE_FREQ}" \
  --steps="${FT_STEPS}" \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size="${FT_BATCH_SIZE}" \
  --policy.optimizer_lr="${FT_LR}" \
  --num_workers="${FT_NUM_WORKERS}" \
  ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"} \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${FT_OUTPUT_DIR}"
