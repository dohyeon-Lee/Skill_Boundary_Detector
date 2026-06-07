#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
# Reference models/configs:
#   base DP cfg  : {project_root}/lerobot/{dp_base_config}
#   DINO source  : {project_root}/{dataset_root}/{dino_source_dataset}_DINO/pg{dino_patch_grid}
# Outputs:
#   prepared DINO: {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
#   DP policy    : {project_root}/DP_outputs/{dp_policy_name}
#   checkpoint   : {project_root}/DP_outputs/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Skip DP train:
#   train_DP: false, or checkpoint already exists at the checkpoint path above
#
# DINO copy only, without submitting DP training:
#   cd {project_root}/lerobot/examples/libero/configs/train_skills/DP
#   DINO_SOURCE_DATASET=libero_90_full_full \
#     {project_root}/.venv/bin/python src/prepare_dino_for_training_dataset.py \
#       --config ../train_skills_config.yaml
#   Use DINO_SOURCE_DATASET only when the source DINO directory should be
#   {dataset_root}/{DINO_SOURCE_DATASET}_DINO instead of the inferred base name.
#
# Submit train_dp_dino.sbatch using Slurm values from train_skills_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
DP_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../train_skills_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

SBATCH_ARGS=(
  --partition="${DP_PARTITION}"
  --qos="${DP_QOS}"
  --gres="${DP_GRES}"
  --cpus-per-task="${DP_CPUS_PER_TASK}"
  --mem="${DP_MEM}"
  --time="${DP_TIME}"
)

if [ -n "${DP_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${DP_NODELIST}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit DP DINO train"
echo "  dataset : ${TARGET_DATASET}"
echo "  output  : ${DP_OUTPUT_DIR}"
echo "  dino    : ${DINO_FEATURE_DIR}"
echo "  slurm   : partition=${DP_PARTITION} qos=${DP_QOS} gres=${DP_GRES}"

echo ""
echo "Prepare target DINO features"
PREPARE_ARGS=(--config "${CONFIG_PATH}")
if [ -n "${TARGET_DATASET}" ]; then
  PREPARE_ARGS+=(--dataset "${TARGET_DATASET}")
fi
"${BOOTSTRAP_PYTHON}" "${DP_SRC_DIR}/prepare_dino_for_training_dataset.py" "${PREPARE_ARGS[@]}"

if [ "${TRAIN_DP}" != "true" ]; then
  echo ""
  echo "Skip DP training because train_DP=false"
  echo "  prepared DINO: ${DINO_FEATURE_DIR}"
  exit 0
fi

if [ -e "${DP_POLICY_PATH}" ]; then
  echo ""
  echo "Skip DP training because checkpoint already exists"
  echo "  checkpoint   : ${DP_POLICY_PATH}"
  echo "  prepared DINO: ${DINO_FEATURE_DIR}"
  exit 0
fi

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${DP_SRC_DIR}/train_dp_dino.sbatch"
