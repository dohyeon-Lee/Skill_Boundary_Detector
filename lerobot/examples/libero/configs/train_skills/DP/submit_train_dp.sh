#!/usr/bin/env bash
# Submit DP (Diffusion Policy) training as a Slurm job — SBD 파이프라인의 Stage 0.
# (구명 submit_train_dp_dino.sh — DINO 인코더/precompute 은퇴로 rename. dp_vision: resnet|state.)
#
# Inputs:
#   dataset name : TRAIN_DATA env or target_dataset in dp_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}   (LeRobot v3)
#   base DP cfg  : {project_root}/lerobot/{dp_base_config} (아키텍처만; 나머지는 런타임 오버라이드)
#   relative     : dp_relative=true면 dataset의 meta/relative_action_stats.json 필요 (ABC build ④-b)
#   EEF relative : dp_eef_relative=true면 derived LIBERO action_contract/stats sidecar 필요
# Outputs:
#   DP policy    : {project_root}/{outputs_root}/DP/{dp_policy_name}
#   checkpoint   : …/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Skip DP train:
#   train_DP: false, or checkpoint already exists at the checkpoint path above
#
# Usage (from this folder):
#   ./submit_train_dp.sh
#   TRAIN_DATA=abc_toy ./submit_train_dp.sh      # dataset 오버라이드

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
DP_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/dp_config.yaml}"

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
if [ -n "${DP_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${DP_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit DP train (vision=${DP_VISION})"
echo "  dataset : ${TARGET_DATASET}"
echo "  output  : ${DP_OUTPUT_DIR}"
echo "  slurm   : partition=${DP_PARTITION} qos=${DP_QOS} gres=${DP_GRES}"

if [ "${TRAIN_DP}" != "true" ]; then
  echo ""
  echo "Skip DP training because train_DP=false"
  exit 0
fi

if [ -e "${DP_POLICY_PATH}" ]; then
  echo ""
  echo "Skip DP training because checkpoint already exists"
  echo "  checkpoint   : ${DP_POLICY_PATH}"
  exit 0
fi

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${DP_SRC_DIR}/train_dp.sbatch"
