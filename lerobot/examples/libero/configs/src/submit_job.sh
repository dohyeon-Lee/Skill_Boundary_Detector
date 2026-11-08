#!/usr/bin/env bash
# Submit a batch script the way this server runs jobs (``scheduler:`` in servers/<name>.yaml):
#   slurm (default)  sbatch, or persistent srun+tmux via slurm_submit_mode
#   local (RunPod)   queue/run scripts on this machine, detached from the terminal
#
# Usage in a submit_*.sh (source it next to snapshot_config.sh), exactly like sbatch:
#   VAR=value submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
#   submit_job_active <id>     # 0 while that job is pending or running (squeue / local record)
# Options must use the --opt=value form (all launchers do). Env assignments before the call reach
# the job either way. SBD_SCHEDULER=slurm|local overrides the server setting.
# SBD_SLURM_SUBMIT_MODE=sbatch|srun|srun_existing overrides the server setting.
# SBD_SRUN_JOB_ID overrides srun_job_id when attaching an additional step.
#
# Local jobs
# * run in the current directory, like Slurm; stdout/stderr follow the script's #SBATCH
#   --output/--error pattern (%x = job name, %j = numeric local job id).
# * get --gres=gpu:N free GPUs as CUDA_VISIBLE_DEVICES; SBD_GPUS=2,3 restricts the pool.
#   Jobs wait for free GPUs without reserving them while pending or waiting for dependencies.
# * --array=0-9%2 (including comma lists/steps), --wrap=COMMAND, --export, afterok / afterany
#   dependencies and %A/%a log names work locally. No fake SLURM_JOB_ID is set.
#   partition, qos, mem, time, ... are ignored; tune them for the Pod separately.
# * are recorded in <repo>/.cache/local_jobs (SBD_LOCAL_JOBS_DIR):
#     bash lerobot/examples/libero/configs/src/submit_job.sh list          # like squeue
#     bash lerobot/examples/libero/configs/src/submit_job.sh stop <id>     # like scancel
# A stopped or crashed job is resumed by running the same submit script again (the train
# scripts resume from their last checkpoint).

_SUBMIT_JOB_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SUBMIT_JOB_REPO="$(cd "${_SUBMIT_JOB_SRC}/../../../../.." && pwd)"

submit_job_scheduler() {
  if [ -n "${SBD_SCHEDULER:-}" ]; then
    echo "${SBD_SCHEDULER}"
    return
  fi
  local python="${_SUBMIT_JOB_REPO}/.venv/bin/python" scheduler
  [ -x "${python}" ] || python=python3
  scheduler="$("${python}" "${_SUBMIT_JOB_SRC}/global_config_loader.py" --key scheduler)" || return 1
  echo "${scheduler:-slurm}"
}

submit_job_slurm_mode() {
  if [ -n "${SBD_SLURM_SUBMIT_MODE:-}" ]; then
    echo "${SBD_SLURM_SUBMIT_MODE}"
    return
  fi
  local python="${_SUBMIT_JOB_REPO}/.venv/bin/python" mode
  [ -x "${python}" ] || python=python3
  mode="$("${python}" "${_SUBMIT_JOB_SRC}/global_config_loader.py" --key slurm_submit_mode)" || return 1
  echo "${mode:-sbatch}"
}

submit_job_srun_job_id() {
  if [ -n "${SBD_SRUN_JOB_ID:-}" ]; then
    echo "${SBD_SRUN_JOB_ID}"
    return
  fi
  local python="${_SUBMIT_JOB_REPO}/.venv/bin/python"
  [ -x "${python}" ] || python=python3
  "${python}" "${_SUBMIT_JOB_SRC}/global_config_loader.py" --key srun_job_id
}

_submit_job_dir() {
  echo "${SBD_LOCAL_JOBS_DIR:-${_SUBMIT_JOB_REPO}/.cache/local_jobs}"
}

# First `#SBATCH --<name>=value` of a script.
_submit_job_directive() {
  sed -n "s/^#SBATCH[[:space:]]\{1,\}--$1=\([^[:space:]]*\).*/\1/p" "$2" | head -n 1
}

_submit_job_alive() {
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null
}

_submit_job_field() {
  sed -n "s/^$1=//p" "$2" | head -n 1
}

_submit_job_pid() {
  cat "$(_submit_job_dir)/$1.pid" 2>/dev/null || true
}

# GPUs held by running (or waiting) local jobs, one per line.
_submit_job_held_gpus() {
  local dir record
  dir="$(_submit_job_dir)"
  for record in "${dir}"/*.job; do
    [ -f "${record}" ] || continue
    _submit_job_alive "$(cat "${record%.job}.pid" 2>/dev/null || true)" || continue
    _submit_job_field gpus "${record}" | tr ',' '\n'
  done | sed '/^$/d'
}

submit_job_active() {
  local id="${1:?submit_job_active needs a job id}" scheduler
  scheduler="$(submit_job_scheduler)" || return 1
  if [ "${scheduler}" = local ]; then
    _submit_job_local_python active "${id}"
    return
  fi
  case "$(squeue -h -j "${id}" -o '%T' 2>/dev/null | head -1 || true)" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING) return 0 ;;
  esac
  return 1
}

_submit_job_local_python() {
  local python="${_SUBMIT_JOB_REPO}/.venv/bin/python"
  [ -x "${python}" ] || python=python3
  "${python}" "${_SUBMIT_JOB_SRC}/local_jobs.py" "$@"
}

_submit_job_local() {
  _submit_job_local_python submit "$@"
}

# Existing raw-sbatch launchers keep their exact Slurm submission behavior.
# Only local servers route these batches through the detached local runner.
submit_batch() {
  local scheduler
  scheduler="$(submit_job_scheduler)" || return 1
  case "${scheduler}" in
    local) _submit_job_local "$@" ;;
    slurm) sbatch "$@" ;;
    *) echo "submit_job: unknown scheduler '${scheduler}' (slurm or local)." >&2; return 2 ;;
  esac
}

submit_job() {
  local scheduler mode python existing_id
  scheduler="$(submit_job_scheduler)" || {
    echo "submit_job: could not read the server's scheduler setting." >&2
    return 1
  }
  case "${scheduler}" in
    slurm)
      mode="$(submit_job_slurm_mode)" || return 1
      case "${mode}" in
        sbatch) sbatch "$@" ;;
        srun|srun_existing)
          python="${_SUBMIT_JOB_REPO}/.venv/bin/python"
          [ -x "${python}" ] || python=python3
          if [ "${mode}" = srun_existing ]; then
            existing_id="$(submit_job_srun_job_id)" || return 1
            "${python}" "${_SUBMIT_JOB_SRC}/srun_session.py" submit-existing "${existing_id}" "$@"
          else
            "${python}" "${_SUBMIT_JOB_SRC}/srun_session.py" submit "$@"
          fi
          ;;
        *) echo "submit_job: unknown slurm_submit_mode '${mode}' (sbatch, srun or srun_existing)." >&2; return 2 ;;
      esac
      ;;
    local) _submit_job_local "$@" ;;
    *) echo "submit_job: unknown scheduler '${scheduler}' (slurm or local)." >&2; return 2 ;;
  esac
}

# The detached local job: record the pid, wait for dependencies, run the script, record its exit.
_submit_job_run() {
  local base="$1" status=0 dep dep_pid
  shift
  echo "$$" >"${base}.pid"
  if [ -n "${SBD_LOCAL_DEPENDENCY:-}" ]; then
    touch "${base}.pending"
    for dep in $(printf '%s' "${SBD_LOCAL_DEPENDENCY#*:}" | tr ':' ' '); do
      dep_pid="$(_submit_job_pid "${dep}")"
      while _submit_job_alive "${dep_pid}"; do sleep "${SBD_LOCAL_DEP_POLL:-10}"; done
      if [ "${SBD_LOCAL_DEPENDENCY%%:*}" = afterok ] \
        && [ "$(cat "$(_submit_job_dir)/${dep}.exit" 2>/dev/null || true)" != 0 ]; then
        echo "submit_job: dependency ${dep} did not finish successfully; not starting." >&2
        rm -f "${base}.pending"
        echo dependency >"${base}.exit"
        return 0
      fi
    done
    rm -f "${base}.pending"
    unset SBD_LOCAL_DEPENDENCY
  fi
  bash "$@" || status=$?
  echo "${status}" >"${base}.exit"
}

_submit_job_state() {
  local base="$1" pid code
  pid="$(cat "${base}.pid" 2>/dev/null || true)"
  if _submit_job_alive "${pid}"; then
    [ -f "${base}.pending" ] && echo pending || echo running
  elif [ -f "${base}.exit" ]; then
    code="$(cat "${base}.exit")"
    case "${code}" in
      0) echo done ;;
      dependency) echo "cancelled(dep)" ;;
      *) echo "failed(${code})" ;;
    esac
  else
    echo stopped
  fi
}

_submit_job_cli() {
  local dir record base
  dir="$(_submit_job_dir)"
  case "${1:-list}" in
    list|stop|active)
      _submit_job_local_python "${@:-list}"
      ;;
    batch)
      shift
      submit_batch "$@"
      ;;
    _run)
      shift
      _submit_job_run "$@"
      ;;
    *)
      echo "usage: bash $0 [list | stop <id>]" >&2
      return 2
      ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  _submit_job_cli "$@"
fi
