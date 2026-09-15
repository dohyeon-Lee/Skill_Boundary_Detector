#!/usr/bin/env bash
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

  if [ ! -d "${shared_dataset_dir}" ]; then
    echo "${label} staging: dataset not found: ${shared_dataset_dir}" >&2
    return 1
  fi
  if [ "$(basename "${shared_dataset_dir}")" != "skillvla" ]; then
    echo "${label} staging: expected a path ending in skillvla: ${shared_dataset_dir}" >&2
    return 1
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "${label} staging: rsync is required." >&2
    return 1
  fi

  run_dir="$(cd "$(dirname "${shared_dataset_dir}")" && pwd)"
  stage_root="${SKILLVLA_LOCAL_STAGE_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
  owner="${USER:-user}"
  job_key="${SLURM_JOB_ID:-$$}"
  stage_base="${stage_root%/}/${owner}_skillvla_${job_key}"
  staged_dataset="${stage_base}/skillvla"

  mkdir -p "${staged_dataset}"
  echo "${label} staging: copying ${shared_dataset_dir} -> ${staged_dataset}" >&2
  rsync -a "${shared_dataset_dir}/" "${staged_dataset}/"

  # Do not enumerate auxiliary filenames here.  In particular, foveated
  # policies need skill_focus_uv.npz, which was absent from the old list.
  # transitions.npz is a large retired Stage-3 artifact; skillvla/ was copied
  # above.  The size cap is a final guard against accidentally staging a new
  # large run-level artifact.
  rsync -a \
    --exclude="transitions.npz" \
    --exclude="skillvla" \
    --max-size=200m \
    "${run_dir}/" "${stage_base}/"

  SKILLVLA_SHARED_DATASET_DIR="${shared_dataset_dir}"
  SKILLVLA_STAGE_BASE="${stage_base}"
  STAGED_SKILLVLA_DATASET_DIR="${staged_dataset}"
  export SKILLVLA_SHARED_DATASET_DIR SKILLVLA_STAGE_BASE STAGED_SKILLVLA_DATASET_DIR
}
