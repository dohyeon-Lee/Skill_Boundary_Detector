#!/usr/bin/env bash
# Block-cyclic pi05 PT (continual-learning curriculum mini-experiment).
#
# Inputs:
#   dataset : {project_root}/{pt_dataset_root}/{pt_dataset}
#   base    : {pi_base}
# Output:
#   model   : {project_root}/{outputs_root}/pi05_cycle_PT/PTcyc_..._g{G}p{K}[_lamX][_bY]
# Config:
#   ./cycle_config.yaml
#
# NOTE: no resume support — an existing output dir is removed and the run starts fresh.
#SBATCH --job-name=pi05_cycPT
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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"             # .../train_pi05_cycle/cycle
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"          # .../train_pi05_cycle
CONFIG_PATH="${TRAIN_PI05_CYCLE_CONFIG:-${SUBMIT_DIR}/cycle_config.yaml}"
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

echo "START pi05 cyclic PT"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "dataset  : ${PT_DATASET_DIR}"
echo "base     : ${PI_BASE}"
echo "output   : ${PT_OUTPUT_DIR}"
echo "lr/bs    : ${PT_LR} / ${PT_BATCH_SIZE}"
echo "cycle    : groups=${CYCLE_N_GROUPS} phase=${CYCLE_PHASE_STEPS} lambda=${CYCLE_DELTA_LAMBDA} beta=${CYCLE_REPTILE_BETA}"
nvidia-smi || true

if [ ! -d "${PT_DATASET_DIR}" ]; then
  echo "Dataset not found: ${PT_DATASET_DIR}" >&2
  exit 1
fi

GPU_FREE_THRESHOLD_MIB=500
echo "Waiting for GPU to be free (threshold: ${GPU_FREE_THRESHOLD_MIB} MiB)..."
while true; do
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  if [ "${USED_MIB}" -le "${GPU_FREE_THRESHOLD_MIB}" ]; then
    echo "GPU is free (${USED_MIB} MiB used). Starting training."
    break
  fi
  echo "  GPU in use: ${USED_MIB} MiB. Waiting 60s..."
  sleep 60
done

if [ -d "${PT_OUTPUT_DIR}" ]; then
  echo "Output dir exists (no resume support). Removing and starting fresh."
  rm -rf "${PT_OUTPUT_DIR}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=1 "${CYCLE_TRAIN_SCRIPT}" \
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
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --save_freq="${PT_SAVE_FREQ}" \
  --log_freq="${PT_LOG_FREQ}" \
  --steps="${PT_STEPS}" \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size="${PT_BATCH_SIZE}" \
  --policy.optimizer_lr="${PT_LR}" \
  --policy.scheduler_decay_lr="${PT_DECAY_LR}" \
  --num_workers="${PT_NUM_WORKERS}" \
  --n_groups="${CYCLE_N_GROUPS}" \
  --phase_steps="${CYCLE_PHASE_STEPS}" \
  --n_cycles="${CYCLE_N_CYCLES}" \
  --group_seed="${CYCLE_GROUP_SEED}" \
  --delta_lambda="${CYCLE_DELTA_LAMBDA}" \
  --delta_max_weight="${CYCLE_DELTA_MAX_WEIGHT}" \
  --reptile_beta="${CYCLE_REPTILE_BETA}" \
  --reptile_beta_end="${CYCLE_REPTILE_BETA_END}" \
  --probe_batches_per_group="${CYCLE_PROBE_BATCHES}" \
  --probe_seed="${CYCLE_PROBE_SEED}" \
  --probe_grad_group="${CYCLE_PROBE_GRAD_GROUP}" \
  --iid_baseline="${CYCLE_IID_BASELINE}"

echo "DONE -> ${PT_OUTPUT_DIR}"
