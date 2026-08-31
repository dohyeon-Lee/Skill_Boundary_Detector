#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset
# Reference models:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Outputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset
#
# Prepare the DP-segmented skillset used by FSQ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
BUILD_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/build_data_config.yaml}"

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
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" \
    --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --stage build_data --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" \
    --config "${CONFIG_PATH}" --stage build_data --shell)"
fi
eval "${RESOLVED_SETTINGS}"

# This build produces only the state-DP skillset. It neither stages visual features nor decodes
# episode videos; FSQ handles its own raw-frame vision inputs later during FSQ training/evaluation.

cd "${SCRIPT_DIR}"
mkdir -p logs "${FSQ_INPUTS_DIR}"

IFS=',' read -r -a PARTITIONS <<< "${SLURM_PARTITIONS}"
IFS=',' read -r -a EXCLUDE_NODES <<< "${SLURM_EXCLUDE_NODES:-}"
FIRST_PART="${PARTITIONS[0]}"
EXCLUDE_STR=$(IFS=,; echo "${EXCLUDE_NODES[*]}")

COMMON_EXPORT="ALL"
COMMON_EXPORT+=",TRAIN_SKILLS_CONFIG=${CONFIG_PATH}"
COMMON_EXPORT+=",TRAIN_DATA=${TARGET_DATASET}"

# ── Curves-only backfill ───────────────────────────────────────────────────────────────────────────
# Add per-episode multimodality (VF cos-divergence) curves to an EXISTING skillset (skills untouched) —
# for skillsets built before curve-dumping. Re-runs the DP over each episode in parallel (same task-shard
# array) dumping only curves/ep*.npz; build_skill_dataset --curves_only --resume skips episodes whose
# curve already exists, so it's safely re-runnable. Usage:
#   CURVES_ONLY=true DP_RUN_NAME=<dp_folder> DP_CHECKPOINT=<ckpt> ./submit_build_data.sh
if [ "${CURVES_ONLY:-false}" = "true" ]; then
  if [ ! -d "${SKILLSET_DIR}/skills" ]; then
    echo "CURVES_ONLY: skillset not found (build it first): ${SKILLSET_DIR}/skills" >&2; exit 1
  fi
  TOTAL_TASKS=$("${BOOTSTRAP_PYTHON}" - <<PY
from pathlib import Path
import pandas as pd
tasks = pd.read_parquet(Path("${RAW_DATASET_DIR}") / "meta" / "tasks.parquet")
print(int(tasks["task_index"].nunique()))
PY
)
  ARRAY_END=$(( (TOTAL_TASKS + SKILLSET_TASKS_PER_JOB - 1) / SKILLSET_TASKS_PER_JOB - 1 ))
  CURVES_ARGS=(
    --partition="${SLURM_PARTITION}"
    --qos="${SLURM_QOS}"
    --gres="${SLURM_GRES}"
    --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
    --mem="${SKILLSET_MEM}"
    --time="${SKILLSET_TIME}"
    --array="0-${ARRAY_END}"
  )
  [ -n "${SLURM_NODELIST}" ]      && CURVES_ARGS+=(--nodelist="${SLURM_NODELIST}")
  [ -n "${SLURM_EXCLUDE_NODES}" ] && CURVES_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
  echo "CURVES_ONLY backfill (skills unchanged) → ${SKILLSET_DIR}/curves"
  echo "  DP    : ${DP_POLICY} ck${DP_CHECKPOINT}  (state-history)"
  echo "  array : 0-${ARRAY_END}  (${TOTAL_TASKS} tasks)"
  CURVES_JOB=$(CURVES_ONLY=true TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
    sbatch --parsable "${CURVES_ARGS[@]}" "${BUILD_SRC_DIR}/build_skillset.sbatch")
  echo "Curves backfill array job: ${CURVES_JOB}"
  [ -n "${PRINT_LAST_JOB:-}" ] && echo "LAST_JOB=${CURVES_JOB}"
  exit 0
fi

if [ -f "${SKILLSET_DONE_PATH}" ] && [ -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset already complete: ${SKILLSET_DONE_PATH}"
elif [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ]; then
  # A global threshold cannot be computed independently by each task-array shard.
  # First collect all curves, reduce them to one threshold, then segment the same
  # task shards from those cached curves without a second DP/VF pass.
  read -r TOTAL_TASKS EXPECTED_EPISODES < <("${BOOTSTRAP_PYTHON}" - <<PY
from pathlib import Path
import pandas as pd
root = Path("${RAW_DATASET_DIR}") / "meta"
tasks = pd.read_parquet(root / "tasks.parquet")
episode_parts = sorted((root / "episodes").rglob("file-*.parquet"))
episodes = pd.concat([pd.read_parquet(path) for path in episode_parts], ignore_index=True)
print(int(tasks["task_index"].nunique()), len(episodes))
PY
)
  ARRAY_END=$(( (TOTAL_TASKS + SKILLSET_TASKS_PER_JOB - 1) / SKILLSET_TASKS_PER_JOB - 1 ))
  ARRAY_ARGS=(
    --partition="${SLURM_PARTITION}"
    --qos="${SLURM_QOS}"
    --gres="${SLURM_GRES}"
    --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
    --mem="${SKILLSET_MEM}"
    --time="${SKILLSET_TIME}"
    --array="0-${ARRAY_END}"
  )
  [ -n "${SLURM_NODELIST}" ] && ARRAY_ARGS+=(--nodelist="${SLURM_NODELIST}")
  [ -n "${SLURM_EXCLUDE_NODES}" ] && ARRAY_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")

  if [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ]; then
    echo "Fixed-global-reference skillset build (curves + cached segmentation)"
    echo "  threshold src : ${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
    if [ ! -f "${SKILLSET_GLOBAL_THRESHOLD_SOURCE}" ]; then
      echo "skillset_global_threshold_source not found: ${SKILLSET_GLOBAL_THRESHOLD_SOURCE}" >&2
      exit 1
    fi
  else
    echo "Global-mean skillset build (two pass)"
  fi
  echo "  skillset      : ${SKILLSET_DIR}"
  echo "  total episodes: ${EXPECTED_EPISODES}"
  echo "  array         : 0-${ARRAY_END}"
  CURVES_JOB=$(CURVES_ONLY=true TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
    sbatch --parsable "${ARRAY_ARGS[@]}" "${BUILD_SRC_DIR}/build_skillset.sbatch")
  echo "Curve collection array job: ${CURVES_JOB}"

  if [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ]; then
    # The source has already been checked above.  Curves are still collected for
    # the target dataset, but segmentation uses the frozen reference threshold.
    THRESHOLD_DEPENDENCY="afterok:${CURVES_JOB}"
  else
    REDUCE_ARGS=(
      --partition="${SLURM_PARTITION}"
      --qos="${SLURM_QOS}"
      --gres="${SLURM_GRES}"
      --cpus-per-task=1
      --mem=2G
      --time=00:10:00
      --dependency="afterok:${CURVES_JOB}"
    )
    [ -n "${SLURM_NODELIST}" ] && REDUCE_ARGS+=(--nodelist="${SLURM_NODELIST}")
    [ -n "${SLURM_EXCLUDE_NODES}" ] && REDUCE_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
    THRESHOLD_JOB=$(EXPECTED_EPISODES="${EXPECTED_EPISODES}" TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
      sbatch --parsable "${REDUCE_ARGS[@]}" "${BUILD_SRC_DIR}/compute_global_boundary_threshold.sbatch")
    echo "Global threshold job: ${THRESHOLD_JOB}"
    THRESHOLD_DEPENDENCY="afterok:${THRESHOLD_JOB}"
  fi

  SEGMENT_ARGS=("${ARRAY_ARGS[@]}" --dependency="${THRESHOLD_DEPENDENCY}")
  SKILLSET_JOB=$(USE_CACHED_CURVES=true TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
    sbatch --parsable "${SEGMENT_ARGS[@]}" "${BUILD_SRC_DIR}/build_skillset.sbatch")
  echo "Cached-curve segmentation array job: ${SKILLSET_JOB}"

  MARK_ARGS=(
    --partition="${SLURM_PARTITION}"
    --qos="${SLURM_QOS}"
    --gres="${SLURM_GRES}"
    --cpus-per-task=1
    --mem=2G
    --time=00:10:00
    --dependency="afterok:${SKILLSET_JOB}"
  )
  [ -n "${SLURM_NODELIST}" ] && MARK_ARGS+=(--nodelist="${SLURM_NODELIST}")
  [ -n "${SLURM_EXCLUDE_NODES}" ] && MARK_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")
  MARK_JOB=$(TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
    sbatch --parsable "${MARK_ARGS[@]}" "${BUILD_SRC_DIR}/mark_skillset_complete.sbatch")
  echo "Skillset marker job: ${MARK_JOB}"
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
    sbatch --parsable "${SKILLSET_ARGS[@]}" "${BUILD_SRC_DIR}/build_skillset.sbatch")
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
    sbatch --parsable "${MARK_ARGS[@]}" "${BUILD_SRC_DIR}/mark_skillset_complete.sbatch")
  echo "Skillset marker job: ${MARK_JOB}"
fi

# Final job a caller (e.g. submit_eval.sh auto-build) should depend on: the skillset marker
# (FSQ visual-token extraction is retired; training/eval decode selected raw frames live.)
LAST_JOB="${MARK_JOB:-}"
[ -n "${LAST_JOB}" ] && echo "  eval can depend on skillset marker job ${LAST_JOB}"
# Final job id on a parseable line so callers can depend on it (omitted when nothing to wait on).
[ -n "${PRINT_LAST_JOB:-}" ] && [ -n "${LAST_JOB}" ] && echo "LAST_JOB=${LAST_JOB}"
