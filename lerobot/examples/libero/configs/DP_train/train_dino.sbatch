#!/usr/bin/env bash
#SBATCH --job-name=DP_DINO_train

# SBATCH --partition=asus_pro6000
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
  CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
fi
if [ ! -f "${CONFIG_PY}" ]; then
  echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
  echo "Submit from lerobot root or from examples/libero/configs/DP_train." >&2
  exit 1
fi
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell --data "${TRAIN_DATA}")"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell)"
fi

cd "${HOMEDIR}${PROJDIR}/lerobot"
source "${HOMEDIR}${PROJDIR}/.venv/bin/activate"

export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${HOMEDIR}/.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${HOMEDIR}/.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: ${SLURMD_NODENAME:-local}"
nvidia-smi

DATASET=${DATA}
DATASET_ROOT=${DATADIR}

N_OBS_STEPS=${DP_N_OBS_STEPS}
N_ACTION_STEPS=${DP_N_ACTION_STEPS}
HORIZON=${DP_HORIZON}

VISUAL_BACKBONE=${IMAGE_FEATURE_TAG}
PATCH_GRID=${PATCH_GRID}
DINO_FEATURE_DIR=${DINO_FEATURE_DIR}
DINO_IMAGE_KEYS="[\"${IMAGE_KEY}\"]"

TRAIN_CONFIG_PATH=${HOMEDIR}${PROJDIR}/lerobot/examples/libero/configs/DP_train/train_dp_${DATASET}.yaml
OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/${DP_POLICY}
WANDB_PROJECT=${WANDB_PROJECT:-DP_train}
JOB_NAME=${JOB_NAME:-${DP_POLICY}}
SAVE_FREQ=${DP_SAVE_FREQ}

BATCH_SIZE=${DP_BATCH_SIZE}
NUM_WORKERS=${DP_NUM_WORKERS}

if [ ! -d "${DINO_FEATURE_DIR}" ]; then
  echo "DINO feature dir not found: ${DINO_FEATURE_DIR}" >&2
  echo "Run configs/data_generation/1_DINO/run_frame_dino_parallel.sh first." >&2
  exit 1
fi

RESUME_FLAG=""
CHECKPOINT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  rm -rf "${OUTPUT_DIR}"
fi

echo "DP DINO train"
echo "  dataset   : ${DATASET}"
echo "  raw data  : ${RAW_DATASET_DIR}"
echo "  dino dir  : ${DINO_FEATURE_DIR}"
echo "  image key : ${IMAGE_KEY}"
echo "  output    : ${OUTPUT_DIR}"
echo "  config    : ${TRAIN_CONFIG_PATH}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  lerobot-train \
  --config_path=${TRAIN_CONFIG_PATH} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --wandb.enable=true \
  --wandb.project=${WANDB_PROJECT} \
  --save_freq=${SAVE_FREQ} \
  --policy.n_obs_steps=${N_OBS_STEPS} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.horizon=${HORIZON} \
  --policy.use_dino_features=true \
  --policy.dino_feature_dir=${DINO_FEATURE_DIR} \
  --policy.dino_image_keys="${DINO_IMAGE_KEYS}" \
  --policy.dino_patch_grid=${PATCH_GRID} \
  --policy.dino_feature_dim=${DP_DINO_FEATURE_DIM} \
  --policy.dino_visual_feature_dim=${DP_DINO_VISUAL_FEATURE_DIM} \
  --policy.dino_transformer_n_layers=${DP_DINO_TRANSFORMER_N_LAYERS} \
  --policy.dino_transformer_n_heads=${DP_DINO_TRANSFORMER_N_HEADS} \
  --dataset.repo_id=lerobot/${DATASET} \
  --dataset.root=${RAW_DATASET_DIR} \
  --dataset.video_backend=pyav \
  --batch_size=${BATCH_SIZE} \
  --num_workers=${NUM_WORKERS} \
  ${RESUME_FLAG}

echo "DONE"
