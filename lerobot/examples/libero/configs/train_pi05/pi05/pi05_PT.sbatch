#!/usr/bin/env bash
# pi05 PT training.
#
# Inputs:
#   dataset : {project_root}/{pt_dataset_root}/{pt_dataset}
#   base    : {pi_base}
# Output:
#   model   : {project_root}/{pi05_outputs_root}/PT_{pt_dataset}_pi05_batch{pt_batch_size}_{pt_exp}
# Config:
#   ./pi05_config.yaml
#SBATCH --job-name=pi05_PT
#SBATCH --partition=debug
#SBATCH --qos=big_qos
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
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"


echo "START pi05 PT"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${PT_DATASET_DIR}"
echo "base     : ${PI_BASE}"
echo "tokenizer: ${PI05_TOKENIZER_PATH}"
echo "output   : ${PT_OUTPUT_DIR}"
echo "lr/bs    : ${PT_LR} / ${PT_BATCH_SIZE}"
echo "lr sched : ${PT_LR_MODE} (warmup=${PT_WARMUP_STEPS})"
echo "freeze   : vision=${PT_FREEZE_VISION_ENCODER} lang=${PT_FREEZE_LANGUAGE_MODEL}"
nvidia-smi || true

if [ ! -d "${PT_DATASET_DIR}" ]; then
  echo "Dataset not found: ${PT_DATASET_DIR}" >&2
  exit 1
fi

CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found. Resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "No checkpoint found. Removing existing output dir and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${PT_NUM_GPUS}" "${TRAIN_BIN}" \
  --dataset.repo_id="lerobot/${PT_DATASET}" \
  --dataset.root="${PT_DATASET_DIR}" \
  --policy.type=pi05 \
  --output_dir="${PT_OUTPUT_DIR}" \
  --job_name="${PT_RUN_NAME}" \
  --policy.pretrained_path="${PI_BASE}" \
  --policy.tokenizer_path="${PI05_TOKENIZER_PATH}" \
  --policy.chunk_size="${PT_CHUNK_SIZE}" \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.project="${PT_WANDB_PROJECT}" \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder="${PT_FREEZE_VISION_ENCODER}" \
  --policy.freeze_language_model="${PT_FREEZE_LANGUAGE_MODEL}" \
  --policy.lora_enable="${PT_LORA_ENABLE}" \
  --policy.lora_rank="${PT_LORA_RANK}" \
  --policy.lora_alpha="${PT_LORA_ALPHA}" \
  --policy.lora_dropout="${PT_LORA_DROPOUT}" \
  --policy.lora_llm="${PT_LORA_LLM}" \
  --policy.lora_vision="${PT_LORA_VISION}" \
  --policy.lora_targets="${PT_LORA_TARGETS}" \
  --policy.push_to_hub=false \
  --save_freq="${PT_SAVE_FREQ}" \
  --steps="${PT_STEPS}" \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size="${PT_BATCH_SIZE}" \
  --policy.optimizer_lr="${PT_LR}" \
  --policy.scheduler_mode="${PT_LR_MODE}" \
  --policy.scheduler_warmup_steps="${PT_WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${PT_DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${PT_DECAY_LR}" \
  --num_workers="${PT_NUM_WORKERS}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
