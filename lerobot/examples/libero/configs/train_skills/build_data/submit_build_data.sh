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
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

# FSQ 학습/평가는 이제 DINO를 ONLINE으로 계산한다 (train_FSQ/fsq_eval이 raw mp4에서 warm-pass —
# 디스크 precompute 완전 제거). 따라서 이 빌드는 FSQ용 per-frame DINO 스테이징도, 토큰 materialize도
# 하지 않는다 — skillset만 만들면 끝. (구 MATERIALIZE_TOKENS/extract_skill_tokens 단계 은퇴.)
#   BUILD_SKILLSET_ONLY=true → 의미 유지 (skillset만; 어차피 그 외 산출물이 없어졌으므로 사실상 동일).

# (DP DINO 스테이징 은퇴 — DP는 resnet(raw frames) / state(proprioceptive)만. dino 인코더 옵션 제거.
#  skill segmentation은 state DP에서 observation.state만 소비 → per-frame DINO 준비가 불필요.)

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
  echo "  DP    : ${DP_POLICY} ck${DP_CHECKPOINT}  (vision=${DP_VISION})"
  echo "  array : 0-${ARRAY_END}  (${TOTAL_TASKS} tasks)"
  CURVES_JOB=$(CURVES_ONLY=true TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" TOTAL_TASKS="${TOTAL_TASKS}" \
    sbatch --parsable "${CURVES_ARGS[@]}" "${BUILD_SRC_DIR}/build_skillset.sbatch")
  echo "Curves backfill array job: ${CURVES_JOB}"
  [ -n "${PRINT_LAST_JOB:-}" ] && echo "LAST_JOB=${CURVES_JOB}"
  exit 0
fi

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
  SKILLSET_DEPENDENCY="--dependency=afterok:${MARK_JOB}"
fi

# Final job a caller (e.g. submit_eval.sh auto-build) should depend on: the skillset marker
# (FSQ visual-token extraction is retired; training/eval decode selected raw frames live.)
LAST_JOB="${MARK_JOB:-}"
[ -n "${LAST_JOB}" ] && echo "  eval can depend on skillset marker job ${LAST_JOB}"
# Final job id on a parseable line so callers can depend on it (omitted when nothing to wait on).
[ -n "${PRINT_LAST_JOB:-}" ] && [ -n "${LAST_JOB}" ] && echo "LAST_JOB=${LAST_JOB}"
