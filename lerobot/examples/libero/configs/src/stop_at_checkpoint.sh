#!/usr/bin/env bash
# Cancel training jobs once they have finished writing a given checkpoint.
#
# Each job is resolved to its output directory from its own Slurm stdout log, so this works for
# any launcher under configs/ that prints an "output : <dir>" line (stage1, NewTask_FT, pi05, ...).
#
# A checkpoint counts as COMPLETE only when checkpoints/last points at it: lerobot-train updates
# that link after the weights and the training state are on disk, so cancelling at that moment
# cannot truncate a checkpoint.
#
# Usage:
#   ./stop_at_checkpoint.sh <step> <jobid> [jobid ...]
#   nohup ./stop_at_checkpoint.sh 100000 2298210 2295065 2298185 > ~/stop_at_100k.log 2>&1 &
#   tail -f ~/stop_at_100k.log
# Env:
#   POLL_SECONDS  seconds between checks (default 300)
#   DRY_RUN=1     log the decision without cancelling anything
set -uo pipefail

TARGET_STEP="${1:-}"
[ -n "${TARGET_STEP}" ] || { echo "usage: $0 <step> <jobid> [jobid ...]" >&2; exit 2; }
shift
[ "$#" -gt 0 ] || { echo "usage: $0 <step> <jobid> [jobid ...]" >&2; exit 2; }
POLL_SECONDS="${POLL_SECONDS:-300}"

log() { echo "[$(date '+%F %T')] $*"; }

output_dir_of() {   # job id -> the output dir its launcher printed
  local stdout
  stdout="$(scontrol show job "$1" 2>/dev/null | tr ' ' '\n' | grep '^StdOut=' | cut -d= -f2-)"
  [ -n "${stdout}" ] && [ -f "${stdout}" ] || return 1
  local dir
  dir="$(grep -m1 -E '^output +:' "${stdout}" | sed -E 's/^output +: *//')"
  [ -n "${dir}" ] || return 1
  printf '%s\n' "${dir}"
}

completed_step_of() {   # output dir -> the step checkpoints/last points at
  local resolved step
  resolved="$(readlink -f "$1/checkpoints/last" 2>/dev/null)" || return 1
  step="$(basename "${resolved}")"
  [[ "${step}" =~ ^[0-9]+$ ]] || return 1
  [ -s "${resolved}/pretrained_model/model.safetensors" ] || return 1
  printf '%s\n' "${step}"
}

log "watching $# job(s) for a completed checkpoint >= ${TARGET_STEP} (every ${POLL_SECONDS}s)${DRY_RUN:+ [DRY_RUN]}"
pending=("$@")
while [ "${#pending[@]}" -gt 0 ]; do
  still=()
  for job in "${pending[@]}"; do
    state="$(squeue -h -j "${job}" -o '%T' 2>/dev/null)"
    if [ -z "${state}" ]; then
      log "job ${job}: no longer queued; stop watching"
      continue
    fi
    if ! out="$(output_dir_of "${job}")"; then
      log "job ${job}: ${state}, output dir not readable yet"
      still+=("${job}")
      continue
    fi
    if step="$(completed_step_of "${out}")" && (( 10#${step} >= 10#${TARGET_STEP} )); then
      if [ "${DRY_RUN:-0}" = 1 ]; then
        log "job ${job}: checkpoint ${step} complete -> would scancel (DRY_RUN)"
        still+=("${job}")
      else
        log "job ${job}: checkpoint ${step} complete -> scancel"
        if scancel "${job}"; then log "job ${job}: cancelled"; else log "job ${job}: scancel FAILED"; fi
      fi
    else
      log "job ${job}: ${state}, at ${step:-<no checkpoint yet>}"
      still+=("${job}")
    fi
  done
  pending=(${still[@]+"${still[@]}"})
  [ "${#pending[@]}" -gt 0 ] || break
  sleep "${POLL_SECONDS}"
done
log "nothing left to watch"
