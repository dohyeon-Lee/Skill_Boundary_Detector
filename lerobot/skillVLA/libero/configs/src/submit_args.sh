#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
# Source before a launcher's own argument parsing. Keep all non-selector args.
# Example: ./submit_train.sh --73428 [existing model/options]
sbd_parse_submit_args() {
  local argument job_id="" options_done=false
  _SBD_LAUNCH_ARGS=()
  for argument in "$@"; do
    if [ "${options_done}" = true ]; then
      _SBD_LAUNCH_ARGS+=("${argument}")
    elif [ "${argument}" = -- ]; then
      options_done=true
      _SBD_LAUNCH_ARGS+=("${argument}")
    elif [[ "${argument}" == --[0-9]* ]]; then
      if [[ ! "${argument}" =~ ^--[1-9][0-9]*$ ]]; then
        echo "Invalid job selector '${argument}'; use --JOB_ID, e.g. --73428." >&2
        return 2
      fi
      if [ -n "${job_id}" ]; then
        echo "Specify only one --JOB_ID." >&2
        return 2
      fi
      job_id="${argument#--}"
    else
      _SBD_LAUNCH_ARGS+=("${argument}")
    fi
  done
  if [ -n "${job_id}" ]; then
    export SBD_SCHEDULER=slurm
    export SBD_SLURM_SUBMIT_MODE=srun_existing
    export SBD_SRUN_JOB_ID="${job_id}"
  fi
}

sbd_parse_submit_args "$@" || return "$?"
set -- "${_SBD_LAUNCH_ARGS[@]}"
unset _SBD_LAUNCH_ARGS
