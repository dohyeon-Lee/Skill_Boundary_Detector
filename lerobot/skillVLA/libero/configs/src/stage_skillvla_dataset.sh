#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
# Stage one SkillVLA run on job-local storage.
#
# The LeRobot dataset is the run's ``skillvla/`` directory, while auxiliary
# artifacts such as skill_initial_state.npz and skill_focus_uv.npz live next
# to it.  Keep the sibling copy generic so newly-added small artifacts do not
# require another hard-coded allow-list update.

stage_skillvla_dataset_on_node() {
  local shared_dataset_dir="${1:?stage_skillvla_dataset_on_node needs a dataset directory}"
  local label="${2:-SkillVLA dataset}"
  local run_dir stage_root owner job_key stage_base staged_dataset
  local -a lock=()
  local -a copy_options=(-a --delete)

  if [ ! -d "${shared_dataset_dir}" ]; then
    echo "${label} staging: dataset not found: ${shared_dataset_dir}" >&2
    return 1
  fi
  if [ "$(basename "${shared_dataset_dir}")" != "skillvla" ]; then
    echo "${label} staging: expected a path ending in skillvla: ${shared_dataset_dir}" >&2
    return 1
  fi
  # A group parent stages before launching concurrent trainers. Workers reuse
  # that completed copy instead of racing rsync --delete against each other.
  if [ "${SKILLVLA_GROUP_STAGE_READY:-0}" = "1" ]; then
    if [ "${SKILLVLA_SHARED_DATASET_DIR:-}" != "${shared_dataset_dir}" ] || \
       [ ! -d "${STAGED_SKILLVLA_DATASET_DIR:-/nonexistent}" ]; then
      echo "${label} staging: group staging is missing or belongs to another dataset." >&2
      return 1
    fi
    echo "${label} staging: reusing group dataset ${STAGED_SKILLVLA_DATASET_DIR}" >&2
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "${label} staging: rsync is required." >&2
    return 1
  fi

  run_dir="$(cd "$(dirname "${shared_dataset_dir}")" && pwd)"
  stage_root="${SKILLVLA_LOCAL_STAGE_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
  owner="${USER:-user}"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    job_key="${SLURM_JOB_ID}"
    if [ -n "${SBD_SRUN_TASK_ID:-}" ]; then
      job_key="${job_key}_${SBD_SRUN_TASK_ID}"
    elif [ "${SBD_SRUN_EXISTING_STEP:-0}" = "1" ]; then
      job_key="${job_key}_step${SLURM_STEP_ID:?Additional srun step requires SLURM_STEP_ID}"
    fi
  else
    # No Slurm (RunPod): nothing clears the stage dir after a run, so keep one copy per dataset
    # and reuse it (rsync then only refreshes changed files) instead of a new copy every run.
    job_key="shared_$(printf '%s' "${run_dir}" | cksum | cut -d' ' -f1)"
  fi
  stage_base="${stage_root%/}/${owner}_skillvla_${job_key}"
  staged_dataset="${stage_base}/skillvla"

  if [ "${SKILLVLA_STAGE_DEREFERENCE_SYMLINKS:-false}" = true ]; then
    # DROID views reference the converted videos with a relative symlink.
    # Materialize that small video corpus on NVMe instead of copying a broken
    # relative link into /tmp or reading remote video for every training batch.
    mkdir -p "${stage_root}"
    local needed available
    needed="$(du -sLB1 "${shared_dataset_dir}" | cut -f1)"
    available="$(df -B1 --output=avail "${stage_root}" | tail -1 | tr -d ' ')"
    if (( available < needed + 1073741824 )); then
      echo "${label} staging: local cache has insufficient room; reading the verified shared dataset" >&2
      SKILLVLA_SHARED_DATASET_DIR="${shared_dataset_dir}"
      STAGED_SKILLVLA_DATASET_DIR="${shared_dataset_dir}"
      export SKILLVLA_SHARED_DATASET_DIR STAGED_SKILLVLA_DATASET_DIR
      return 0
    fi
    copy_options=(-aL --delete)
  fi
  mkdir -p "${staged_dataset}"
  if [ -z "${SLURM_JOB_ID:-}" ] && command -v flock >/dev/null 2>&1; then
    lock=(flock "${stage_base}.lock")      # two runs on one dataset: the second waits, then syncs nothing
  fi
  echo "${label} staging: copying ${shared_dataset_dir} -> ${staged_dataset}" >&2
  ${lock[@]+"${lock[@]}"} rsync "${copy_options[@]}" "${shared_dataset_dir}/" "${staged_dataset}/"

  # Do not enumerate auxiliary filenames here.  In particular, foveated
  # policies need skill_focus_uv.npz, which was absent from the old list.
  # transitions.npz is a large retired Stage-3 artifact; skillvla/ was copied
  # above.  The size cap is a final guard against accidentally staging a new
  # large run-level artifact.
  ${lock[@]+"${lock[@]}"} rsync -a \
    --exclude="transitions.npz" \
    --exclude="skillvla" \
    --max-size=200m \
    "${run_dir}/" "${stage_base}/"

  SKILLVLA_SHARED_DATASET_DIR="${shared_dataset_dir}"
  SKILLVLA_STAGE_BASE="${stage_base}"
  STAGED_SKILLVLA_DATASET_DIR="${staged_dataset}"
  export SKILLVLA_SHARED_DATASET_DIR SKILLVLA_STAGE_BASE STAGED_SKILLVLA_DATASET_DIR
}
