#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
# Reference models:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Outputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset
#
# Submit DP-based skillset generation using the shared train_skills_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSQ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON_SRC_DIR="${FSQ_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${FSQ_DIR}/../train_skills_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

TOTAL_TASKS=$("${BOOTSTRAP_PYTHON}" - <<PY
from pathlib import Path
import pandas as pd
tasks = pd.read_parquet(Path("${RAW_DATASET_DIR}") / "meta" / "tasks.parquet")
print(int(tasks["task_index"].nunique()))
PY
)

ARRAY_END=$(( (TOTAL_TASKS + SKILLSET_TASKS_PER_JOB - 1) / SKILLSET_TASKS_PER_JOB - 1 ))

SBATCH_ARGS=(
  --partition="${SLURM_PARTITION}"
  --qos="${SLURM_QOS}"
  --gres="${SLURM_GRES}"
  --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
  --mem="${SKILLSET_MEM}"
  --time="${SKILLSET_TIME}"
  --array="0-${ARRAY_END}"
)

if [ -n "${SLURM_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${SLURM_NODELIST}")
fi
if [ -n "${SLURM_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit skillset generation"
echo "  dataset       : ${TARGET_DATASET}"
echo "  raw data      : ${RAW_DATASET_DIR}"
echo "  DP policy     : ${DP_POLICY_PATH}"
echo "  DINO          : ${DINO_FEATURE_DIR}"
echo "  output        : ${SKILLSET_DIR}"
echo "  total tasks   : ${TOTAL_TASKS}"
echo "  tasks/job     : ${SKILLSET_TASKS_PER_JOB}"
echo "  array         : 0-${ARRAY_END}"

SKILLSET_JOB=$(TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
  sbatch --parsable "${SBATCH_ARGS[@]}" "${FSQ_SRC_DIR}/build_skillset.sbatch")
echo "Skillset array job: ${SKILLSET_JOB}"

MARK_ARGS=(
  --partition="${SLURM_PARTITION}"
  --qos="${SLURM_QOS}"
  --gres="${SLURM_GRES}"  # QOSMinGRES: 이 클러스터는 모든 job에 GPU >=1 요구
  --cpus-per-task=1
  --mem=2G
  --time=00:10:00
  --dependency="afterok:${SKILLSET_JOB}"
)
if [ -n "${SLURM_NODELIST}" ]; then
  MARK_ARGS+=(--nodelist="${SLURM_NODELIST}")
fi
if [ -n "${SLURM_EXCLUDE_NODES}" ]; then
  MARK_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

MARK_JOB=$(TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch --parsable "${MARK_ARGS[@]}" "${FSQ_SRC_DIR}/mark_skillset_complete.sbatch")
echo "Skillset marker job: ${MARK_JOB}"
