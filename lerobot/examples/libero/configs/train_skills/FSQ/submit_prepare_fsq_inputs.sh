#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   frame DINO   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset
# Reference models:
#   DP policy    : {project_root}/DP_outputs/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
#   SAM2         : sam2_checkpoint or {project_root}/models/sam2/sam2.1_hiera_large.pt
# Outputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#   SAM2 flags   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/patch_flags.npz
#
# Prepare FSQ inputs from skillset + prepared frame DINO.
# Produces skill-level DINO tokens and, in dino_flags mode, SAM2 patch flags.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../train_skills_config.yaml}"
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

if [ ! -d "${DINO_FEATURE_DIR}" ]; then
  echo "Prepared frame DINO not found: ${DINO_FEATURE_DIR}" >&2
  echo "Run DP/submit_train_dp_dino.sh first." >&2
  exit 1
fi

BUILD_PATCH_FLAGS=false
if [ "${FSQ_BUILD_PATCH_FLAGS}" = "true" ]; then
  BUILD_PATCH_FLAGS=true
fi
if ${BUILD_PATCH_FLAGS} && [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "SAM2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
  exit 1
fi

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
echo "  tokens       : ${DINO_TOKENS_PATH}"
echo "  patch flags  : ${FSQ_BUILD_PATCH_FLAGS}"
if ${BUILD_PATCH_FLAGS}; then
  echo "  sam2 masks   : ${SAM2_MASKS_DIR}"
  echo "  sam2 flags   : ${SAM2_FLAGS_PATH}"
fi

EXTRACT_JOB=$(sbatch --parsable \
  --partition="${FIRST_PART}" \
  --qos="${SLURM_QOS}" \
  ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
  ${SKILLSET_DEPENDENCY:+${SKILLSET_DEPENDENCY}} \
  --export="${COMMON_EXPORT}" \
  "${FSQ_SRC_DIR}/extract_skill_tokens.sbatch")
echo "Skill token extraction job: ${EXTRACT_JOB}"

if ! ${BUILD_PATCH_FLAGS}; then
  echo "SAM2 patch flags skipped (fsq_build_patch_flags=false)"
  exit 0
fi

declare -A NODE_PARTITION
declare -A NODE_TOTAL_GPU
for part in "${PARTITIONS[@]}"; do
  [ -z "${part}" ] && continue
  while read -r node gres; do
    [ -z "${node}" ] && continue
    total=$(echo "${gres}" | sed -nE 's/.*gpu(:[^:[:space:]]*)?:([0-9]+).*/\2/p' | head -1)
    if [ -z "${total}" ] || [ "${total}" -eq 0 ]; then
      continue
    fi
    [ "${NODE_PARTITION[$node]+_}" ] && continue
    for ex in "${EXCLUDE_NODES[@]}"; do
      [ -n "${ex}" ] && [ "${node}" = "${ex}" ] && continue 2
    done
    NODE_PARTITION[$node]=${part}
    NODE_TOTAL_GPU[$node]=${total}
  done < <(sinfo -p "${part}" -N -t idle,alloc,mix -o "%N %G" --noheader 2>/dev/null)
done

N_WORKERS=0
for node in "${!NODE_TOTAL_GPU[@]}"; do
  total=${NODE_TOTAL_GPU[$node]}
  w=$(( total - SLURM_GPU_RESERVE ))
  [ "${w}" -lt 0 ] && w=0
  [ "${w}" -gt "${SLURM_GPU_MAX_PER_NODE}" ] && w=${SLURM_GPU_MAX_PER_NODE}
  N_WORKERS=$(( N_WORKERS + w ))
done

if [ "${N_WORKERS}" -eq 0 ]; then
  echo "ERROR: No GPU nodes found in partitions: ${PARTITIONS[*]}" >&2
  exit 1
fi
if [ "${N_WORKERS}" -gt "${FSQ_PRECOMPUTE_MAX_WORKERS}" ]; then
  N_WORKERS=${FSQ_PRECOMPUTE_MAX_WORKERS}
fi

PARTITIONS_STR=$(IFS=,; echo "${PARTITIONS[*]}")
SAM2_EXPORT="${COMMON_EXPORT},N_WORKERS=${N_WORKERS}"
RECOVERY_WORKERS=${FSQ_PRECOMPUTE_RECOVERY_WORKERS}

SAM2_JOB=$(sbatch --parsable \
  --partition="${PARTITIONS_STR}" \
  --qos="${SLURM_QOS}" \
  ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
  --array="0-$(( N_WORKERS - 1 ))" \
  ${SKILLSET_DEPENDENCY:+${SKILLSET_DEPENDENCY}} \
  --export="${SAM2_EXPORT}" \
  "${FSQ_SRC_DIR}/precompute_sam2_masks_worker.sbatch")
echo "SAM2 mask workers job: ${SAM2_JOB} (array 0-$(( N_WORKERS - 1 )))"

RECOVERY_EXPORT="${COMMON_EXPORT},N_WORKERS=${RECOVERY_WORKERS}"
RECOVERY_JOB=$(sbatch --parsable \
  --partition="${PARTITIONS_STR}" \
  --qos="${SLURM_QOS}" \
  ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
  --array="0-$(( RECOVERY_WORKERS - 1 ))" \
  --dependency="afterany:${SAM2_JOB}" \
  --export="${RECOVERY_EXPORT}" \
  "${FSQ_SRC_DIR}/precompute_sam2_masks_worker.sbatch")
echo "SAM2 recovery job: ${RECOVERY_JOB}"

MERGE_JOB=$(sbatch --parsable \
  --partition="${PARTITIONS_STR}" \
  --qos="${SLURM_QOS}" \
  ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
  --dependency="afterok:${RECOVERY_JOB}" \
  --export="${COMMON_EXPORT}" \
  "${FSQ_SRC_DIR}/merge_sam2_patch_flags.sbatch")
echo "SAM2 merge job: ${MERGE_JOB}"

if [ "${FSQ_CLEANUP_SAM2_MASKS}" = "true" ]; then
  CLEANUP_JOB=$(sbatch --parsable \
    --partition="${FIRST_PART}" \
    --qos="${SLURM_QOS}" \
    ${EXCLUDE_STR:+--exclude="${EXCLUDE_STR}"} \
    --dependency="afterok:${MERGE_JOB}" \
    --export="${COMMON_EXPORT}" \
    "${FSQ_SRC_DIR}/cleanup_sam2_masks.sbatch")
  echo "SAM2 cleanup job: ${CLEANUP_JOB}"
fi
