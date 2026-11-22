#!/usr/bin/env bash
# Shared early dispatch. Sourcing workflow helpers keeps their original behavior.
if [[ "${BASH_SOURCE[1]:-}" == "$0" ]]; then
  _sbd_yonsei_requested=false
  for _sbd_remote_arg in "$@"; do
    [[ "${_sbd_remote_arg}" != -- ]] || break
    case "${_sbd_remote_arg}" in
      --yonsei|--yonsei-plan|--yonsei-check) _sbd_yonsei_requested=true ;;
    esac
  done
  if [[ "${_sbd_yonsei_requested}" == true ]]; then
    _sbd_remote_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    _sbd_remote_python="${_sbd_remote_root}/.venv/bin/python"
    [[ -x "${_sbd_remote_python}" ]] || _sbd_remote_python=python3
    exec "${_sbd_remote_python}" "${_sbd_remote_root}/tools/remote/yonsei.py" \
      launch --script "${BASH_SOURCE[1]}" -- "$@"
  fi
  unset _sbd_yonsei_requested _sbd_remote_arg
fi
