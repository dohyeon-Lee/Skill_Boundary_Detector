#!/usr/bin/env bash
# Requeue a job that never starts making progress on its node.
#
# gpu_guard.sh catches a node whose CUDA is broken, but a GPU-free job never
# touches CUDA and so has no health check at all. Such a job can hang before its
# first log line -- on 2026-08-20 every fsq_gt_replay task that landed on node17
# or node49 hung that way (6/6, versus 24/24 finishing elsewhere) and burned its
# whole 30-minute limit without replaying a single occurrence.
#
# This guard watches the command's stderr for a marker the work prints once it is
# past start-up. If the marker has not appeared within the timeout, the node is
# recorded and the job is requeued without it -- the same recovery gpu_guard.sh
# performs.
#
# Usage, in place of running the command directly:
#
#     source "${CONFIGS_ROOT}/stall_guard.sh"
#     run_with_startup_guard 600 "episode source" -- python worker.py --flag
#
# The sbatch also needs `#SBATCH --requeue` for the retry path to be allowed.
#
# Cost: one shell loop that stats a file every STALL_GUARD_POLL seconds and
# stops as soon as the marker appears, so it adds nothing measurable to the run.
#
# Environment:
#   STALL_GUARD_POLL      seconds between checks (default 5)
#   STALL_GUARD_DISABLE=1 run the command unguarded

STALL_GUARD_POLL="${STALL_GUARD_POLL:-5}"

# Kill the command and everything it spawned. Signalling only the direct child
# would leave the real worker running: it would keep writing into the output
# directory and race the requeued attempt for the same files.
_stall_guard_kill_tree() {
  local pid="$1"
  local signal="${2:-TERM}"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null); do
    _stall_guard_kill_tree "${child}" "${signal}"
  done
  kill "-${signal}" "${pid}" 2>/dev/null || true
}

run_with_startup_guard() {
  local timeout="${1:?run_with_startup_guard needs a timeout in seconds}"; shift
  local marker="${1:?run_with_startup_guard needs a progress marker}"; shift
  [ "${1:-}" != "--" ] || shift
  [ "$#" -gt 0 ] || { echo "STALL GUARD: no command given." >&2; return 2; }

  if [ "${STALL_GUARD_DISABLE:-0}" = "1" ]; then
    "$@"
    return
  fi

  local log
  log="$(mktemp "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/stall_guard.XXXXXX.log")"
  # Duplicate stderr rather than piping it: the command keeps writing to the
  # job's own stderr file, and its exit status stays this function's to return.
  "$@" 2> >(tee -a "${log}" >&2) &
  local pid=$!

  local waited=0
  while [ "${waited}" -lt "${timeout}" ]; do
    kill -0 "${pid}" 2>/dev/null || break
    ! grep -qF -- "${marker}" "${log}" 2>/dev/null || break
    sleep "${STALL_GUARD_POLL}"
    waited=$((waited + STALL_GUARD_POLL))
  done

  if kill -0 "${pid}" 2>/dev/null && ! grep -qF -- "${marker}" "${log}" 2>/dev/null; then
    echo "STALL GUARD: no '${marker}' after ${timeout}s; treating this node as stuck." >&2
    _stall_guard_kill_tree "${pid}" TERM
    sleep 2
    _stall_guard_kill_tree "${pid}" KILL
    rm -f "${log}"
    # Sourced from the same directory; gpu_guard.sh owns the shared recovery.
    if ! declare -F guard_requeue_current_job >/dev/null; then
      source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"
    fi
    guard_requeue_current_job "STALL GUARD" \
      "the job produced no progress within ${timeout}s" \
      "Exclude the node and resubmit."
    return 1
  fi

  local status=0
  wait "${pid}" || status=$?
  rm -f "${log}"
  return "${status}"
}
