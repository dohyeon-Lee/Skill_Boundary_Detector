#!/usr/bin/env bash
# Shared CUDA preflight for every GPU job in this project.
#
# A node can present a healthy GPU in nvidia-smi and still fail torch's CUDA
# initialization ("CUDA unknown error"). Without a check the run silently drops
# to CPU: orders of magnitude slower, still holding the GPU allocation, and
# usually noticed only hours later. node26 in global_config.yaml is the
# documented example, and node04/05/08/10/12/13/14/16/18/19/21/23/24/31/39
# have shown the same behaviour.
#
# On a bad node this records the hostname and asks Slurm to requeue the job with
# that node excluded, so the work lands on a healthy GPU instead of dying or
# crawling on CPU.
#
# Usage, right after the `nvidia-smi` line of an sbatch script:
#
#     source "${CONFIGS_ROOT}/gpu_guard.sh"
#     require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
#
# A Python entrypoint that supports the inline guard can avoid the separate
# cold ``import torch`` above:
#
#     prepare_inline_cuda_guard
#     if python_entrypoint ...; then status=0; else status=$?; fi
#     handle_inline_cuda_guard_exit "${status}"
#     (( status == 0 )) || exit "${status}"
#
# The sbatch also needs `#SBATCH --requeue` for the retry path to be allowed.
#
# Environment:
#   GPU_GUARD_BAD_NODE_FILE  append-only log of nodes that failed (default:
#                            ${PROJECT_ROOT}/.slurm_bad_gpu_nodes)
#   GPU_GUARD_MAX_REQUEUE    give up after this many retries (default 4)
#   LEROBOT_ALLOW_CPU_FALLBACK=1  skip the check and permit CPU, matching the
#                            escape hatch honoured by lerobot/configs/policies.py

GPU_GUARD_BAD_NODE_FILE="${GPU_GUARD_BAD_NODE_FILE:-${PROJECT_ROOT:-${HOME}}/.slurm_bad_gpu_nodes}"
GPU_GUARD_MAX_REQUEUE="${GPU_GUARD_MAX_REQUEUE:-4}"
GPU_GUARD_INLINE_EXIT_CODE=86

# Record the current node and requeue this job without it. Shared with
# stall_guard.sh: both guards diagnose "this node cannot run the work", and the
# recovery -- log the node, grow ExcNodeList, requeue -- is identical.
#   $1 label shown in the messages (e.g. "GPU GUARD")
#   $2 what went wrong, for the log line
#   $3 what happens if we cannot requeue, for the log line
guard_requeue_current_job() {
  local label="${1:-GUARD}"
  local reason="${2:-this node cannot run the job}"
  local giving_up="${3:-Check the cluster, then resubmit.}"
  local node="${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
  local attempt="${SLURM_RESTART_COUNT:-0}"
  # For one task per array SLURM_JOB_ID equals the array-parent id, and
  # scontrol update on the parent silently fails to pin ExcNodeList to the
  # task. The <array>_<task> form is unambiguous for every task.
  local jobref="${SLURM_JOB_ID:-}"
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    jobref="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  echo "${label}: ${reason} on ${node} (attempt ${attempt})." >&2

  # Recorded so submit scripts can exclude the node up front next time.
  printf '%s\t%s\tjob=%s\n' "$(date -Is)" "${node}" "${SLURM_JOB_ID:-none}" \
    >>"${GPU_GUARD_BAD_NODE_FILE}" 2>/dev/null || true

  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "${label}: not running under Slurm; cannot requeue." >&2
    exit 1
  fi
  if [ "${attempt}" -ge "${GPU_GUARD_MAX_REQUEUE}" ]; then
    echo "${label}: already requeued ${attempt} times; giving up." "${giving_up}" >&2
    exit 1
  fi

  local exclude
  exclude="$(scontrol show job "${jobref}" 2>/dev/null |
    tr ' ' '\n' | sed -n 's/^ExcNodeList=//p' | head -1)"
  case "${exclude}" in
    "" | "(null)") exclude="${node}" ;;
    *) exclude="${exclude},${node}" ;;
  esac

  local updated=1
  scontrol update JobId="${jobref}" ExcNodeList="${exclude}" 2>/dev/null || updated=0

  # Survive the SIGTERM Slurm sends while requeueing us, so the post-requeue
  # retry below still runs on sites that refuse updates on running jobs.
  trap '' TERM
  if scontrol requeue "${jobref}" 2>/dev/null; then
    if [ "${updated}" = "0" ]; then
      scontrol update JobId="${jobref}" ExcNodeList="${exclude}" 2>/dev/null || true
    fi
    echo "${label}: requeued ${jobref} excluding ${exclude}." >&2
  else
    echo "${label}: requeue was refused (is '#SBATCH --requeue' set?)." "${giving_up}" >&2
  fi
  exit 1
}

_gpu_guard_requeue_current_job() {
  guard_requeue_current_job "GPU GUARD" \
    "torch cannot initialize CUDA" \
    "Exiting instead of running on CPU."
}

require_cuda_or_requeue() {
  local python_bin="${1:?require_cuda_or_requeue needs a python interpreter}"

  if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
    echo "GPU GUARD: LEROBOT_ALLOW_CPU_FALLBACK=1, skipping the CUDA check." >&2
    return 0
  fi
  if [ ! -x "${python_bin}" ]; then
    echo "GPU GUARD: ${python_bin} is not executable; skipping the CUDA check." >&2
    return 0
  fi
  if "${python_bin}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    return 0
  fi
  _gpu_guard_requeue_current_job
}

prepare_inline_cuda_guard() {
  if [ "${LEROBOT_ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
    export LEROBOT_INLINE_CUDA_GUARD=0
    unset LEROBOT_CUDA_GUARD_FAILURE_MARKER
    echo "GPU GUARD: LEROBOT_ALLOW_CPU_FALLBACK=1, skipping the CUDA check." >&2
    return 0
  fi

  local marker_root="${SLURM_TMPDIR:-/tmp}"
  export LEROBOT_INLINE_CUDA_GUARD=1
  export LEROBOT_CUDA_GUARD_FAILURE_MARKER="${marker_root}/lerobot_cuda_guard_${SLURM_JOB_ID:-$$}.failed"
  rm -f "${LEROBOT_CUDA_GUARD_FAILURE_MARKER}"
}

handle_inline_cuda_guard_exit() {
  local status="${1:?handle_inline_cuda_guard_exit needs the entrypoint status}"
  if [ "${LEROBOT_INLINE_CUDA_GUARD:-0}" = "1" ] && \
     [ -f "${LEROBOT_CUDA_GUARD_FAILURE_MARKER:-/nonexistent}" ]; then
    if [ "${status}" -ne "${GPU_GUARD_INLINE_EXIT_CODE}" ]; then
      echo "GPU GUARD: CUDA failure marker found (entrypoint status=${status})." >&2
    fi
    _gpu_guard_requeue_current_job
  fi
}
