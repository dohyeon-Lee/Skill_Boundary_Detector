#!/usr/bin/env bash
# Replay-free FT of a cycle-PT checkpoint + live PT-group forgetting probes.
#
# Inputs:
#   source : {ft_source.model_dir}/checkpoints/{ft_source_checkpoint}/pretrained_model
#   data   : {project_root}/{dataset_root}/{ft_dataset}  (새 task)
#   probes : {pt_dataset} + source run의 groups.json (PT 때와 동일 probe)
# Output:
#   model  : {outputs_root}/pi05_cycle_FT/FTcyc_{label}_{ckpt}_{ft_dataset}_batch{bs}
# Config:
#   ./ft_config.yaml
#
# NOTE: no resume — an existing output dir is removed and the run starts fresh.
#SBATCH --job-name=cycFT
#SBATCH --partition=debug
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"             # .../train_pi05_cycle/cycle_ft
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"          # .../train_pi05_cycle
CONFIG_PATH="${TRAIN_PI05_CYCLE_FT_CONFIG:-${SUBMIT_DIR}/ft_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_cycle_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START cycle FT (+probes)"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "source   : ${FT_PRETRAINED_MODEL_PATH}"
echo "dataset  : ${FT_DATASET_DIR}"
echo "probes   : ${PT_DATASET_DIR} groups=${FT_GROUPS_JSON} every=${FT_PROBE_EVERY}"
echo "output   : ${FT_OUTPUT_DIR}"
nvidia-smi || true

for d in "${FT_DATASET_DIR}" "${PT_DATASET_DIR}"; do
  [ -d "$d" ] || { echo "Dataset not found: $d" >&2; exit 1; }
done
[ -d "${FT_PRETRAINED_MODEL_PATH}" ] || { echo "Source ckpt not found: ${FT_PRETRAINED_MODEL_PATH}" >&2; exit 1; }
[ -f "${FT_GROUPS_JSON}" ] || { echo "groups.json not found: ${FT_GROUPS_JSON} (set ft_groups_json)" >&2; exit 1; }

GPU_FREE_THRESHOLD_MIB=500
echo "Waiting for GPU to be free (threshold: ${GPU_FREE_THRESHOLD_MIB} MiB)..."
while true; do
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  if [ "${USED_MIB}" -le "${GPU_FREE_THRESHOLD_MIB}" ]; then
    echo "GPU is free (${USED_MIB} MiB used). Starting."
    break
  fi
  echo "  GPU in use: ${USED_MIB} MiB. Waiting 60s..."
  sleep 60
done

if [ -d "${FT_OUTPUT_DIR}" ]; then
  echo "Output dir exists (no resume support). Removing and starting fresh."
  rm -rf "${FT_OUTPUT_DIR}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=1 \
  "${ROOT_DIR}/src/lerobot_train_ft_probe.py" \
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
  --policy.train_expert_only="${FT_TRAIN_EXPERT_ONLY}" \
  --policy.push_to_hub=false \
  --save_freq="${FT_SAVE_FREQ}" \
  --log_freq="${FT_LOG_FREQ}" \
  --steps="${FT_STEPS}" \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size="${FT_BATCH_SIZE}" \
  --policy.optimizer_lr="${FT_LR}" \
  --num_workers="${FT_NUM_WORKERS}" \
  --probe_dataset_repo_id="lerobot/${PT_DATASET}" \
  --probe_dataset_root="${PT_DATASET_DIR}" \
  --probe_groups_json="${FT_GROUPS_JSON}" \
  --probe_every="${FT_PROBE_EVERY}" \
  --probe_batches_per_group="${CYCLE_PROBE_BATCHES}" \
  --probe_seed="${CYCLE_PROBE_SEED}"

echo "DONE -> ${FT_OUTPUT_DIR}"
