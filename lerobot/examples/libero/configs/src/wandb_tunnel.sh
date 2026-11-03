#!/usr/bin/env bash
# Source from runtime_env.sh. No root, extra Python dependencies, or global DNS edits.

sbd_clear_stale_batch_proxies() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  [ -n "${PROJECT_ROOT:-}" ] || return 0
  local helper="${PROJECT_ROOT}/lerobot/examples/libero/configs/src/batch_proxy_env.py"
  [ -f "${helper}" ] || return 0
  local python_bin="${PROJECT_ROOT}/.venv/bin/python" names name
  [ -x "${python_bin}" ] || python_bin="$(command -v python3)"
  names="$("${python_bin}" "${helper}")" || return 1
  for name in ${names}; do
    case "${name}" in
      HTTP_PROXY|http_proxy|HTTPS_PROXY|https_proxy|ALL_PROXY|all_proxy|WANDB_HTTP_PROXY|WANDB_HTTPS_PROXY)
        unset "${name}" ;;
      *) return 1 ;;
    esac
  done
  [ -z "${names}" ] || echo "Cleared unavailable inherited localhost proxy variables: ${names}"
}

sbd_start_wandb_tunnel() {
  local mode="${SBD_WANDB_TUNNEL:-auto}"
  local node="${SLURMD_NODENAME:-$(hostname -s)}"
  case "${mode}" in
    0|off|false) return 0 ;;
    auto)
      [ -n "${SLURM_JOB_ID:-}" ] && [ "${node%%.*}" = node100 ] || return 0
      ;;
    1|on|true) ;;
    *) echo 'W&B tunnel: SBD_WANDB_TUNNEL must be auto, on, or off.' >&2; return 1 ;;
  esac
  case "${WANDB_MODE:-online}" in offline|disabled) return 0 ;; esac
  case "${WANDB_ENABLE:-true}:${WANDB_DISABLED:-false}" in
    false:*|0:*|*:true|*:1) return 0 ;;
  esac
  # Re-sourcing the shared runtime in the same job must not create more helpers.
  if [ "${SBD_WANDB_TUNNEL_OWNER:-}" = "${BASHPID}" ] && \
     [ -n "${SBD_WANDB_TUNNEL_PID:-}" ] && kill -0 "${SBD_WANDB_TUNNEL_PID}" 2>/dev/null; then
    return 0
  fi
  local python_bin="${PROJECT_ROOT}/.venv/bin/python"
  [ -x "${python_bin}" ] || python_bin="$(command -v python3)"
  local helper="${PROJECT_ROOT}/lerobot/examples/libero/configs/src/wandb_ssh_proxy.py"
  local directory pid ready values owner_pid="${BASHPID}"
  directory="$(mktemp -d /tmp/sbd-wandb-XXXXXXXX)" || return 1
  "${python_bin}" "${helper}" --host "${SBD_WANDB_RELAY_HOST:-node01}" \
    --directory "${directory}" --owner-pid "${owner_pid}" \
    >"${directory}/log" 2>&1 &
  pid=$!
  ready="${directory}/ready.json"
  local attempt
  for ((attempt=0; attempt<350; attempt++)); do
    [ -f "${ready}" ] && break
    kill -0 "${pid}" 2>/dev/null || break
    sleep .1
  done
  if [ ! -f "${ready}" ]; then
    echo 'W&B tunnel failed: check passwordless SSH and known_hosts for the relay node.' >&2
    [ ! -f "${directory}/log" ] || tail -n 8 "${directory}/log" >&2
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    rm -rf -- "${directory}"
    return 1
  fi
  values="$("${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["proxy"])' "${ready}")" || return 1
  export WANDB_HTTP_PROXY="${values}" WANDB_HTTPS_PROXY="${values}"
  # The Python API uses requests' HTTPS proxy; the Go logging backend also
  # needs W&B's settings above. WANDB__PROXIES is not parsed by W&B 0.24.
  # Only this job inherits the relay (including its other HTTPS requests).
  export HTTPS_PROXY="${values}" https_proxy="${values}"
  export SBD_WANDB_TUNNEL_PID="${pid}"
  export SBD_WANDB_TUNNEL_OWNER="${owner_pid}"
  echo "W&B tunnel: ${node%%.*} -> ${SBD_WANDB_RELAY_HOST:-node01} -> W&B (online)."
  # The helper watches this shell's PID/start time and also handles Slurm SIGTERM.
  # No EXIT/TERM traps are replaced, including existing CUDA/staging guard traps.
}

sbd_clear_stale_batch_proxies
sbd_start_wandb_tunnel
