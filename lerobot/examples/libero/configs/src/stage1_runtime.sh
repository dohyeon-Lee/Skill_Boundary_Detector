#!/usr/bin/env bash
# Runtime helpers for the Stage-1 launcher; sourcing alone has no side effects.

stage1_set_cpu_threads() {
  local count="${STAGE1_CPU_THREADS:-1}"
  if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'STAGE1_CPU_THREADS must be a positive integer.' >&2
    return 1
  fi
  export OMP_NUM_THREADS="${count}" MKL_NUM_THREADS="${count}"
  export OPENBLAS_NUM_THREADS="${count}" NUMEXPR_NUM_THREADS="${count}"
}

stage1_configure_launcher() {
  local gpus="${1:?GPU count required}" entrypoint="${2:?Trainer required}"
  if [[ ! "${gpus}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Stage-1 GPU count must be a positive integer.' >&2
    return 1
  fi
  TRAIN_LAUNCH=("${entrypoint}")
  if (( gpus > 1 )); then
    TRAIN_LAUNCH=(accelerate launch --num_processes="${gpus}" "${entrypoint}")
  fi
}

stage1_cleanup_runtime() {
  local status=$? pid="${SBD_WANDB_TUNNEL_PID:-}" attempt parent command
  trap - EXIT
  # Only stop this shell's own proxy, never a group parent's or a reused PID.
  if [ "${SBD_WANDB_TUNNEL_OWNER:-}" = "${BASHPID}" ] && \
     [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && [ -r "/proc/${pid}/cmdline" ]; then
    parent="$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null || true)"
    command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [ "${parent}" = "${BASHPID}" ] && [[ "${command}" == *wandb_ssh_proxy.py* ]]; then
      kill "${pid}" 2>/dev/null || true
      for ((attempt=0; attempt<50; attempt++)); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep .1
      done
      kill -KILL "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
  exit "${status}"
}
