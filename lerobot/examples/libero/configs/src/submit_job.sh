#!/usr/bin/env bash
# Submit a batch script the way this server runs jobs (``scheduler:`` in servers/<name>.yaml):
#   slurm (default)  sbatch, unchanged
#   local (RunPod)   run the script in the background right away, detached from the terminal
#
# Usage in a submit_*.sh (source it next to snapshot_config.sh), exactly like sbatch:
#   VAR=value submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
#   submit_job_active <id>     # 0 while that job is pending or running (squeue / local record)
# Options must use the --opt=value form (all launchers do). Env assignments before the call reach
# the job either way. SBD_SCHEDULER=slurm|local overrides the server setting.
#
# Local jobs
# * run in the current directory, like Slurm; stdout/stderr follow the script's #SBATCH
#   --output/--error pattern (%x = job name, %j = local job id, a number like 260922101500).
# * get --gres=gpu:N free GPUs (not held by another running or waiting local job) as
#   CUDA_VISIBLE_DEVICES; SBD_GPUS=2,3 picks them by hand. With too few free GPUs nothing
#   starts (there is no queue).
# * --dependency=afterok:<id>[:<id>] / afterany:... waits for those local jobs (holding its GPUs);
#   after a failed afterok dependency the job does not start. --array and --wrap are refused;
#   partition, qos, mem, time, ... are ignored.
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
    _submit_job_alive "$(_submit_job_pid "${id}")"
    return
  fi
  case "$(squeue -h -j "${id}" -o '%T' 2>/dev/null | head -1 || true)" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING) return 0 ;;
  esac
  return 1
}

_submit_job_local() {
  local name="" output="" error="" gres="" dependency="" parsable=0 script=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --array*|--wrap*)
        echo "submit_job: ${1%%=*} needs Slurm; this server runs jobs locally (scheduler: local)." >&2
        return 2 ;;
      --dependency=*) dependency="${1#*=}" ;;
      --parsable) parsable=1 ;;
      --job-name=*) name="${1#*=}" ;;
      --output=*) output="${1#*=}" ;;
      --error=*) error="${1#*=}" ;;
      --gres=*) gres="${1#*=}" ;;
      -*) ;;                                   # Slurm resource/placement options: not used locally
      *) script="$1"; shift; break ;;
    esac
    shift
  done
  if [ -z "${script}" ] || [ ! -f "${script}" ]; then
    echo "submit_job: batch script not found: ${script:-<none>}" >&2
    return 2
  fi

  local dir dep
  dir="$(_submit_job_dir)"
  if [ -n "${dependency}" ]; then
    case "${dependency}" in
      *,*|*\?*) dependency="unsupported" ;;
    esac
    case "${dependency}" in
      afterok:?*|afterany:?*) ;;
      *)
        echo "submit_job: only --dependency=afterok:<id>[:<id>] or afterany:... work locally." >&2
        return 2 ;;
    esac
    for dep in $(printf '%s' "${dependency#*:}" | tr ':' ' '); do
      if [ ! -f "${dir}/${dep}.job" ]; then
        echo "submit_job: dependency ${dep} is not a local job." >&2
        return 2
      fi
    done
  fi

  name="${name:-$(_submit_job_directive job-name "${script}")}"
  name="${name:-$(basename "${script}")}"
  output="${output:-$(_submit_job_directive output "${script}")}"
  error="${error:-$(_submit_job_directive error "${script}")}"
  gres="${gres:-$(_submit_job_directive gres "${script}")}"
  output="${output:-slurm-%j.out}"
  error="${error:-${output}}"

  local want=0
  case "${gres}" in
    gpu*) want="${gres##*:}"; [[ "${want}" =~ ^[0-9]+$ ]] || want=1 ;;
  esac

  local id lock
  mkdir -p "${dir}"
  exec {lock}>"${dir}/.lock"
  if command -v flock >/dev/null 2>&1; then flock "${lock}"; fi

  id="$(date +%y%m%d%H%M%S)"
  while [ -e "${dir}/${id}.job" ]; do id=$((id + 1)); done

  local gpus=""
  if [ "${want}" -gt 0 ] && [ -n "${SBD_GPUS:-}" ]; then
    gpus="${SBD_GPUS}"
  elif [ "${want}" -gt 0 ]; then
    local all held free
    if ! all="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
      echo "submit_job: nvidia-smi failed; cannot pick GPUs (set SBD_GPUS=0,1,... to choose)." >&2
      exec {lock}>&-
      return 1
    fi
    held="$(_submit_job_held_gpus)"
    free="$(printf '%s\n' "${all}" | tr -d ' ' | grep -vxF -f <(printf '%s\n' "${held:-none}") || true)"
    if [ "$(printf '%s' "${free}" | grep -c .)" -lt "${want}" ]; then
      echo "submit_job: ${name} needs ${want} GPU(s) but only $(printf '%s' "${free}" | grep -c .) are free." >&2
      echo "  running local jobs:  bash ${_SUBMIT_JOB_SRC}/submit_job.sh list" >&2
      exec {lock}>&-
      return 1
    fi
    gpus="$(printf '%s\n' "${free}" | head -n "${want}" | paste -sd, -)"
  fi

  output="$(printf '%s' "${output}" | sed "s/%x/${name}/g; s/%j/${id}/g")"
  error="$(printf '%s' "${error}" | sed "s/%x/${name}/g; s/%j/${id}/g")"
  mkdir -p "$(dirname "${output}")" "$(dirname "${error}")"
  output="$(cd "$(dirname "${output}")" && pwd)/$(basename "${output}")"
  error="$(cd "$(dirname "${error}")" && pwd)/$(basename "${error}")"

  local base="${dir}/${id}"
  {
    echo "name=${name}"
    echo "script=${script}"
    echo "cwd=${PWD}"
    echo "gpus=${gpus}"
    echo "dependency=${dependency}"
    echo "output=${output}"
    echo "error=${error}"
    echo "started=$(date -Is)"
  } >"${base}.job"

  # setsid: own session, so closing the terminal does not stop it and `stop` can signal the whole
  # process group (accelerate workers, dataloader workers).
  (
    # Inherit no open files (e.g. a lock the submit script holds): the job would keep it locked.
    for fd in /proc/"${BASHPID}"/fd/*; do
      fd="${fd##*/}"
      if [[ "${fd}" =~ ^[0-9]+$ ]] && [ "${fd}" -gt 2 ]; then eval "exec ${fd}>&-" 2>/dev/null || true; fi
    done
    if [ -n "${gpus}" ]; then export CUDA_VISIBLE_DEVICES="${gpus}"; fi
    if [ -n "${dependency}" ]; then export SBD_LOCAL_DEPENDENCY="${dependency}"; fi
    export SBD_LOCAL_JOB_ID="${id}"
    setsid nohup bash "${_SUBMIT_JOB_SRC}/submit_job.sh" _run "${base}" "${script}" "$@" \
      >"${output}" 2>"${error}" </dev/null &
  )
  local waited=0
  while [ ! -s "${base}.pid" ] && [ "${waited}" -lt 50 ]; do sleep 0.1; waited=$((waited + 1)); done
  exec {lock}>&-

  if [ "${parsable}" = 1 ]; then
    echo "${id}"
  else
    echo "Submitted local job ${id} (pid $(cat "${base}.pid" 2>/dev/null || echo '?'), GPU ${gpus:-none})"
    [ -z "${dependency}" ] || echo "  waits for: ${dependency}"
    echo "  log : ${output}"
    [ "${error}" = "${output}" ] || echo "  err : ${error}"
    echo "  list: bash ${_SUBMIT_JOB_SRC}/submit_job.sh list    stop: bash ${_SUBMIT_JOB_SRC}/submit_job.sh stop ${id}"
  fi
}

submit_job() {
  local scheduler
  scheduler="$(submit_job_scheduler)" || {
    echo "submit_job: could not read the server's scheduler setting." >&2
    return 1
  }
  case "${scheduler}" in
    slurm) sbatch "$@" ;;
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
    list)
      printf '%-13s %-15s %-6s %-26s %s\n' ID STATE GPUS STARTED NAME
      for record in "${dir}"/*.job; do
        [ -f "${record}" ] || continue
        base="${record%.job}"
        printf '%-13s %-15s %-6s %-26s %s\n' "$(basename "${base}")" "$(_submit_job_state "${base}")" \
          "$(_submit_job_field gpus "${record}")" "$(_submit_job_field started "${record}")" \
          "$(_submit_job_field name "${record}")  $(_submit_job_field output "${record}")"
      done
      ;;
    stop)
      base="${dir}/${2:?usage: submit_job.sh stop <id>}"
      [ -f "${base}.job" ] || { echo "No local job ${2}." >&2; return 1; }
      local pid
      pid="$(cat "${base}.pid" 2>/dev/null || true)"
      if ! _submit_job_alive "${pid}"; then
        echo "Local job ${2} is not running ($(_submit_job_state "${base}"))."
        return 0
      fi
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}"
      echo "Sent SIGTERM to local job ${2} (process group ${pid})."
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
