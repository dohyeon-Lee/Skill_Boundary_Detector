#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
# Build ABC DP curves, one global threshold, and cached skill segmentation.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../src/submit_args.sh"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ABC_BUILD_DATA_CONFIG:-${SCRIPT_DIR}/build_data_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../src/train_skills_config.py" \
  --config "${CONFIG_PATH}" --stage build_data --shell)"
eval "${RESOLVED_SETTINGS}"
export ABC_BUILD_DATA_CONFIG="${CONFIG_PATH}"

cd "${SCRIPT_DIR}"
mkdir -p logs "${FSQ_INPUTS_DIR}"

if [ -f "${SKILLSET_DONE_PATH}" ]; then
  echo "ABC skillset already complete: ${SKILLSET_DONE_PATH}"
  exit 0
fi
if [ ! -d "${DP_POLICY_PATH}" ]; then
  echo "ABC DP checkpoint not found: ${DP_POLICY_PATH}" >&2
  echo "Train DP through checkpoint ${DP_CHECKPOINT} before building skill data." >&2
  exit 1
fi

SHARD_COUNT=$(( (ABC_TASK_COUNT + SKILLSET_TASKS_PER_JOB - 1) / SKILLSET_TASKS_PER_JOB ))
ARRAY_END=$(( SHARD_COUNT - 1 ))
GPU_ARGS=(
  --partition="${SLURM_PARTITION}"
  --qos="${SLURM_QOS}"
  --gres="${SLURM_GRES}"
  --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
  --mem="${SKILLSET_MEM}"
  --time="${SKILLSET_TIME}"
)
[ -n "${SLURM_NODELIST}" ] && GPU_ARGS+=(--nodelist="${SLURM_NODELIST}")
[ -n "${SLURM_EXCLUDE_NODES}" ] && GPU_ARGS+=(--exclude="${SLURM_EXCLUDE_NODES}")

echo "Submit ABC skill-boundary build"
echo "  dataset    : ${RAW_DATASET_DIR}"
echo "  source task: ${ABC_SOURCE_TASKS_CSV} (${ABC_EXPECTED_EPISODES} episodes)"
echo "  prompt IDs : ${ABC_TASK_INDICES}"
echo "  DP         : ${DP_POLICY_PATH}"
echo "  output     : ${SKILLSET_DIR}"
echo "  detector   : ${SKILLSET_MODE}, ${SKILLSET_ACTION_MODE} 14D, continuous grippers [6,13]"
echo "  SBD profile: ${SKILLSET_SBD_PROFILE} (power=${SKILLSET_SCORE_POWER}, SG=${SKILLSET_SMOOTH_WINDOW}/${SKILLSET_SAVGOL_POLYORDER})"
echo "  timing     : replan=${SKILLSET_REPLAN_INTERVAL}, nms=${SKILLSET_NMS_DIST}, terminal=${SKILLSET_TERMINAL_MASK_MODE}/pre${SKILLSET_TERMINAL_RELEASE_PRE_OFFSET_FRAMES} @ 30 Hz"

export ABC_EXPECTED_EPISODES ABC_TASK_INDICES_JSON ABC_SOURCE_TASKS_CSV
CURVES_ONLY=true
export CURVES_ONLY
if [ "${SKILLSET_SINGLE_GPU_WORKERS}" -gt 0 ]; then
  echo "  GPU mode   : one shared GPU, ${SKILLSET_SINGLE_GPU_WORKERS} workers; episodes split automatically"
  if [ "$(submit_job_scheduler)" = slurm ] && [ "$(submit_job_slurm_mode)" != sbatch ]; then
    # This is one process supervising every pipeline stage, so it needs no
    # numeric batch job ID, array, or scheduler dependency. In particular,
    # --JOB_ID must dispatch inside the existing allocation without --parsable.
    ABC_SINGLE_GPU_BUILD=true submit_job \
      --output=logs/%x_%j.out --error=logs/%x_%j.err \
      "${GPU_ARGS[@]}" "${SCRIPT_DIR}/src/build_skillset.sbatch"
  else
    CURVES_JOB="$(ABC_SINGLE_GPU_BUILD=true submit_job --parsable \
      --output=logs/%x_%j.out --error=logs/%x_%j.err \
      "${GPU_ARGS[@]}" "${SCRIPT_DIR}/src/build_skillset.sbatch")"
    echo "  pipeline job    : ${CURVES_JOB} (curves -> threshold -> cached segment -> verify)"
  fi
  echo "The one allocation writes .complete only after all ${ABC_EXPECTED_EPISODES} episodes are verified."
  exit 0
else
  GPU_JOB_COUNT=$(( (SHARD_COUNT + SKILLSET_WORKERS_PER_GPU - 1) / SKILLSET_WORKERS_PER_GPU ))
  GPU_ARRAY_END=$(( GPU_JOB_COUNT - 1 ))
  echo "  GPU mode   : Slurm array, ${GPU_JOB_COUNT} GPUs maximum, ${SKILLSET_WORKERS_PER_GPU} worker(s)/GPU"
  echo "               ${SHARD_COUNT} logical shards (${SKILLSET_TASKS_PER_JOB} prompts/shard)"
  export ABC_LOGICAL_SHARD_COUNT="${SHARD_COUNT}"
  if [ "${SKILLSET_WORKERS_PER_GPU}" -gt 1 ]; then
    CURVES_JOB="$(ABC_ARRAY_GPU_BUILD=true submit_job --parsable --array="0-${GPU_ARRAY_END}" \
      "${GPU_ARGS[@]}" "${SCRIPT_DIR}/src/build_skillset.sbatch")"
  else
    CURVES_JOB="$(submit_job --parsable --array="0-${GPU_ARRAY_END}" \
      "${GPU_ARGS[@]}" "${SCRIPT_DIR}/src/build_skillset.sbatch")"
  fi
fi
echo "  curve job       : ${CURVES_JOB}"

REDUCE_ARGS=(
  --partition="${SKILLSET_CPU_PARTITION}"
  --qos="${SKILLSET_CPU_QOS}"
  --cpus-per-task=1 --mem=2G --time=00:20:00
  --dependency="afterok:${CURVES_JOB}"
)
THRESHOLD_JOB="$(submit_job --parsable "${REDUCE_ARGS[@]}" \
  "${SCRIPT_DIR}/src/compute_global_boundary_threshold.sbatch")"
echo "  threshold job   : ${THRESHOLD_JOB}"

unset CURVES_ONLY
USE_CACHED_CURVES=true
export USE_CACHED_CURVES
SEGMENT_ARGS=(
  --job-name=ABC_segment_cached
  --partition="${SKILLSET_CPU_PARTITION}"
  --qos="${SKILLSET_CPU_QOS}"
  --cpus-per-task="${SKILLSET_CPUS_PER_TASK}"
  --mem="${SKILLSET_MEM}"
  --time="${SKILLSET_TIME}"
  --array="0-${ARRAY_END}"
  --dependency="afterok:${THRESHOLD_JOB}"
)
SEGMENT_JOB="$(submit_job --parsable "${SEGMENT_ARGS[@]}" "${SCRIPT_DIR}/src/build_skillset.sbatch")"
echo "  segmentation job: ${SEGMENT_JOB}"

unset USE_CACHED_CURVES
MARK_ARGS=(
  --partition="${SKILLSET_CPU_PARTITION}"
  --qos="${SKILLSET_CPU_QOS}"
  --cpus-per-task=1 --mem=2G --time=00:10:00
  --dependency="afterok:${SEGMENT_JOB}"
)
MARK_JOB="$(submit_job --parsable "${MARK_ARGS[@]}" \
  "${SCRIPT_DIR}/src/mark_skillset_complete.sbatch")"
echo "  completion job  : ${MARK_JOB}"
echo "The .complete marker is written only after all ${ABC_EXPECTED_EPISODES} episodes are verified."
