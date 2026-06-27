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

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START pi05 PT"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${PT_DATASET_DIR}"
echo "base     : ${PI_BASE}"
echo "output   : ${PT_OUTPUT_DIR}"
echo "lr/bs    : ${PT_LR} / ${PT_BATCH_SIZE}"
nvidia-smi || true

if [ ! -d "${PT_DATASET_DIR}" ]; then
  echo "Dataset not found: ${PT_DATASET_DIR}" >&2
  exit 1
fi

# GPU selection.
# IMPORTANT: under Slurm, gres already pins a dedicated GPU via CUDA_VISIBLE_DEVICES. Overriding
# it (scanning nvidia-smi and picking the "emptiest" GPU) desyncs Slurm's accounting: Slurm still
# thinks our reserved GPU is in use and hands the GPU we actually grabbed to another job, so two
# jobs collide on one physical GPU -> CUDA "unknown error" crashes. So under Slurm we TRUST Slurm's
# allocation and never rescan. Only in a bare foreground run (no Slurm) do we pick the emptiest GPU.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "Under Slurm (job ${SLURM_JOB_ID}); using Slurm-allocated CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'. Starting training."
else
  GPU_FREE_THRESHOLD_MIB=500
  NEED_GPUS="${PT_NUM_GPUS:-1}"
  echo "No Slurm detected; selecting ${NEED_GPUS} emptiest GPU(s) (threshold: ${GPU_FREE_THRESHOLD_MIB} MiB)..."
  while true; do
    # index,used sorted by used-memory ascending; the N emptiest are the first N rows.
    SORTED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2,2 -n)
    SEL_IDX=$(echo "${SORTED}" | head -n "${NEED_GPUS}" | awk -F',' '{gsub(/ /,"",$1); print $1}' | paste -sd, -)
    MAX_USED=$(echo "${SORTED}" | head -n "${NEED_GPUS}" | awk -F',' '{gsub(/ /,"",$2); print $2}' | sort -n | tail -n1)
    if [ "${MAX_USED:-999999}" -le "${GPU_FREE_THRESHOLD_MIB}" ]; then
      export CUDA_VISIBLE_DEVICES="${SEL_IDX}"
      echo "Selected GPU(s) ${CUDA_VISIBLE_DEVICES} (max ${MAX_USED} MiB used). Starting training."
      break
    fi
    echo "  Emptiest ${NEED_GPUS} GPU(s) still busy (max ${MAX_USED} MiB @ idx ${SEL_IDX}). Waiting 60s..."
    sleep 60
  done
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
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --wandb.project="${PT_WANDB_PROJECT}" \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder="${PT_FREEZE_VISION_ENCODER}" \
  --policy.train_expert_only="${PT_TRAIN_EXPERT_ONLY}" \
  --policy.freeze_language_model="${PT_FREEZE_LANGUAGE_MODEL}" \
  --policy.probe_freeze="${PT_PROBE_FREEZE}" \
  --policy.push_to_hub=false \
  --save_freq="${PT_SAVE_FREQ}" \
  --steps="${PT_STEPS}" \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size="${PT_BATCH_SIZE}" \
  --policy.optimizer_lr="${PT_LR}" \
  --num_workers="${PT_NUM_WORKERS}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "DONE -> ${PT_OUTPUT_DIR}"
