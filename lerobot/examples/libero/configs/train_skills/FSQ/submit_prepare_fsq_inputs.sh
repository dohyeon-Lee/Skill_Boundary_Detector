#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
#                  (없으면 {dataset_root}/{dataset}_DINO/pg*에서 자동 준비 — DP 학습 불필요)
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset
# Reference models:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Outputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset
#   DINO tokens  : {.../FSQ_inputs/dino_tokens_pg{grid}.npz, dino_tokens_wrist_pg{grid}.npz}
#
# Prepare FSQ inputs from skillset + prepared frame DINO.
# Produces skill-level DINO tokens for BOTH cameras (3rd-person + wrist) — the FSQ
# terminator reads both via separate DINO-token encoders.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}/src"
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

# Ensure prepared frame DINO exists for BOTH grids: dp (skillset build) and fsq (token
# extraction). When fsq_patch_grid == dino_patch_grid the two dirs are identical → 2nd is a no-op.
ensure_dino() {  # $1 = variant (dp|fsq), $2 = expected dir
  local variant="$1" dir="$2"
  [ -d "${dir}" ] && return 0
  echo "Prepared frame DINO (${variant}) not found → 자동 준비 (DINO 배치만, DP 학습 없음): ${dir}"
  local args=(--config "${CONFIG_PATH}" --variant "${variant}")
  [ -n "${TARGET_DATASET}" ] && args+=(--dataset "${TARGET_DATASET}")
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../DP/src/prepare_dino_for_training_dataset.py" "${args[@]}"
  [ -d "${dir}" ] || { echo "DINO 준비 실패: ${dir}" >&2; exit 1; }
}
ensure_dino dp  "${DINO_FEATURE_DIR}"
ensure_dino fsq "${FSQ_DINO_FEATURE_DIR}"

cd "${SCRIPT_DIR}"
mkdir -p logs "${FSQ_INPUTS_DIR}"

IFS=',' read -r -a PARTITIONS <<< "${SLURM_PARTITIONS}"
IFS=',' read -r -a EXCLUDE_NODES <<< "${SLURM_EXCLUDE_NODES:-}"
FIRST_PART="${PARTITIONS[0]}"
EXCLUDE_STR=$(IFS=,; echo "${EXCLUDE_NODES[*]}")

COMMON_EXPORT="ALL"
COMMON_EXPORT+=",TRAIN_SKILLS_CONFIG=${CONFIG_PATH}"
COMMON_EXPORT+=",TRAIN_DATA=${TARGET_DATASET}"

SKILLSET_DEPENDENCY=""
if [ -f "${SKILLSET_DONE_PATH}" ] && [ -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset already complete: ${SKILLSET_DONE_PATH}"
else
  TOTAL_TASKS=$("${BOOTSTRAP_PYTHON}" - <<PY
from pathlib import Path
import pandas as pd
tasks = pd.read_parquet(Path("${RAW_DATASET_DIR}") / "meta" / "tasks.parquet")
print(int(tasks["task_index"].nunique()))
PY
)
  ARRAY_END=$(( (TOTAL_TASKS + SKILLSET_TASKS_PER_JOB - 1) / SKILLSET_TASKS_PER_JOB - 1 ))

  SKILLSET_ARGS=(
    --partition="${SLURM_PARTITION}"
    --qos="${SLURM_QOS}"
    --gres="${SLURM_GRES}"
    --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
    --mem="${SKILLSET_MEM}"
    --time="${SKILLSET_TIME}"
    --array="0-${ARRAY_END}"
  )
  if [ -n "${SLURM_NODELIST}" ]; then
    SKILLSET_ARGS+=(--nodelist="${SLURM_NODELIST}")
  fi
  if [ -n "${SLURM_EXCLUDE_NODES}" ]; then
    SKILLSET_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
  fi

  echo "Skillset not marked complete; submitting skillset generation first."
  echo "  skillset      : ${SKILLSET_DIR}"
  echo "  total tasks   : ${TOTAL_TASKS}"
  echo "  array         : 0-${ARRAY_END}"
  SKILLSET_JOB=$(TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
    sbatch --parsable "${SKILLSET_ARGS[@]}" "${FSQ_SRC_DIR}/build_skillset.sbatch")
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
  SKILLSET_DEPENDENCY="--dependency=afterok:${MARK_JOB}"
fi

echo "Prepare FSQ inputs"
echo "  dataset      : ${TARGET_DATASET}"
echo "  skillset     : ${SKILLSET_DIR}/skills"
echo "  frame DINO   : ${DINO_FEATURE_DIR}"
echo "  tokens 3rd   : ${DINO_TOKENS_PATH}"
echo "  tokens wrist : ${DINO_TOKENS_WRIST_PATH}"

EXTRACT_JOB=$(sbatch --parsable \
  --partition="${FIRST_PART}" \
  --qos="${SLURM_QOS}" \
  ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
  ${SKILLSET_DEPENDENCY:+${SKILLSET_DEPENDENCY}} \
  --export="${COMMON_EXPORT}" \
  "${FSQ_SRC_DIR}/extract_skill_tokens.sbatch")
echo "Skill token extraction job (both cameras): ${EXTRACT_JOB}"
