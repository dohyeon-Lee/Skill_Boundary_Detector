#!/usr/bin/env bash
# Package a shared project venv into one sequential read, then unpack it onto
# a compute node's local scratch. This avoids thousands of Lustre metadata
# operations during large PyTorch imports.

prepare_node_local_venv_archive() {
  local project_root="${1:?prepare_node_local_venv_archive needs PROJECT_ROOT}"
  local label="${2:-Node-local venv}"
  local source_venv
  source_venv="$(readlink -f "${project_root}/.venv")"
  if [ ! -x "${source_venv}/bin/python" ]; then
    echo "${label}: shared interpreter is missing: ${source_venv}/bin/python" >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "${label}: zstd/flock unavailable; using the shared venv." >&2
    return 1
  fi

  local archive_root="${NODE_LOCAL_VENV_ARCHIVE_DIR:-${project_root}/.cache/node_local_venv}"
  mkdir -p "${archive_root}"

  # Include dependency manifests plus top-level installed-package metadata.
  # The latter catches local `pip install` changes even when a lock file was
  # not updated, without walking every file in the multi-gigabyte venv.
  local fingerprint
  fingerprint="$({
    printf '%s\n' "${source_venv}"
    sha256sum "${source_venv}/pyvenv.cfg"
    local manifest
    for manifest in \
      "${project_root}/requirements.txt" \
      "${project_root}/pyproject.toml" \
      "${project_root}/uv.lock" \
      "${project_root}/lerobot/pyproject.toml" \
      "${project_root}/lerobot/uv.lock" \
      "${project_root}/lerobot/requirements-ubuntu.txt"; do
      if [ -f "${manifest}" ]; then
        sha256sum "${manifest}"
      fi
    done
    local package_root
    for package_root in "${source_venv}"/lib/python*/site-packages; do
      if [ -d "${package_root}" ]; then
        LC_ALL=C find "${package_root}" -mindepth 1 -maxdepth 1 \
          -printf '%f|%y|%s|%T@|%l\n' | LC_ALL=C sort
      fi
    done
    LC_ALL=C find "${source_venv}/bin" -mindepth 1 -maxdepth 1 \
      -printf '%f|%y|%s|%T@|%l\n' | LC_ALL=C sort
  } | sha256sum | cut -d' ' -f1)" || return 1

  local archive="${archive_root}/venv-${fingerprint:0:16}.tar.zst"
  local size_file="${archive}.unpacked_size"
  local lock_file="${archive}.lock"
  if ! (
    flock 9
    if [ ! -s "${archive}" ]; then
      local temporary="${archive}.tmp.$$"
      trap 'rm -f -- "${temporary}"' EXIT
      echo "${label}: creating one-time archive ${archive}" >&2
      if ! (set -o pipefail; tar -C "${source_venv}" -cf - . | \
        zstd -1 -T"${NODE_LOCAL_VENV_ARCHIVE_THREADS:-4}" -q -o "${temporary}"); then
        echo "${label}: archive creation failed; using the shared venv." >&2
        exit 1
      fi
      if ! mv "${temporary}" "${archive}"; then
        echo "${label}: could not publish archive; using the shared venv." >&2
        exit 1
      fi
      trap - EXIT
      echo "${label}: archive ready." >&2
    fi
    if [ ! -s "${size_file}" ]; then
      local size_tmp="${size_file}.tmp.$$"
      du -s --apparent-size --block-size=1 "${source_venv}" \
        | awk '{print $1}' > "${size_tmp}"
      mv "${size_tmp}" "${size_file}"
    fi
  ) 9>"${lock_file}"; then
    return 1
  fi
  printf '%s\n' "${archive}"
}


stage_node_local_venv() {
  local archive="${1:?stage_node_local_venv needs an archive}"
  local shared_venv="${2:?stage_node_local_venv needs a fallback venv}"
  local label="${3:-Node-local venv}"
  if [ ! -s "${archive}" ]; then
    echo "${label}: archive unavailable; using ${shared_venv}." >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "${label}: zstd/flock unavailable on compute node; using ${shared_venv}." >&2
    return 1
  fi

  local owner="${USER:-}"
  [ -n "${owner}" ] || owner="$(id -un)"
  local cache_root="${NODE_LOCAL_VENV_ROOT:-}"
  # Prefer a host-wide local cache so array tasks sharing one node also share
  # one extraction. This cluster does not define SLURM_TMPDIR.
  if [ -z "${cache_root}" ] && [ -d /dev/shm ] && [ -w /dev/shm ]; then
    cache_root="/dev/shm/${owner}/node_local_venv"
  elif [ -z "${cache_root}" ] && [ -n "${SLURM_TMPDIR:-}" ] \
    && [ -d "${SLURM_TMPDIR}" ]; then
    cache_root="${SLURM_TMPDIR}/node_local_venv"
  elif [ -z "${cache_root}" ] && [ -d /tmp ] && [ -w /tmp ]; then
    cache_root="/tmp/${owner}/node_local_venv"
  fi
  if [ -z "${cache_root}" ] || [[ "${cache_root}" != /* ]]; then
    echo "${label}: no writable node-local cache root; using ${shared_venv}." >&2
    return 1
  fi
  if ! mkdir -p "${cache_root}"; then
    echo "${label}: cannot create ${cache_root}; using ${shared_venv}." >&2
    return 1
  fi

  local archive_name fingerprint
  archive_name="$(basename "${archive}")"
  fingerprint="${archive_name%.tar.zst}"
  case "${fingerprint}" in
    *[!A-Za-z0-9._-]*)
      echo "${label}: unsafe archive name ${archive_name}; using ${shared_venv}." >&2
      return 1
      ;;
  esac
  local final_root="${cache_root}/${fingerprint}"
  local lock_file="${cache_root}/.${fingerprint}.lock"

  (
    flock -x 9
    if [ -x "${final_root}/bin/python" ] \
      && [ -f "${final_root}/.node_local_venv_archive" ] \
      && [ "$(cat "${final_root}/.node_local_venv_archive")" = "${archive_name}" ]; then
      echo "${label}: reusing ${final_root}." >&2
      printf '%s\n' "${final_root}"
      exit 0
    fi

    local expected_bytes
    if [ -s "${archive}.unpacked_size" ]; then
      expected_bytes="$(tr -dc '0-9' < "${archive}.unpacked_size")"
    else
      expected_bytes=$(( $(stat -c %s "${archive}") * 3 ))
    fi
    local available_bytes reserve_bytes
    available_bytes="$(df -P --block-size=1 "${cache_root}" | awk 'NR == 2 {print $4}')"
    reserve_bytes=$(( ${NODE_LOCAL_VENV_RESERVE_GB:-8} * 1024 * 1024 * 1024 ))
    if [ -z "${expected_bytes}" ] \
      || (( expected_bytes > available_bytes \
        || available_bytes - expected_bytes < reserve_bytes )); then
      echo "${label}: insufficient space in ${cache_root}; using ${shared_venv}." >&2
      exit 1
    fi

    if [ -e "${final_root}" ]; then
      local invalid_root="${cache_root}/.invalid"
      mkdir -p "${invalid_root}"
      mv "${final_root}" "${invalid_root}/${fingerprint}.$(date +%s).$$"
    fi
    local partial_root="${cache_root}/.${fingerprint}.partial.$$"
    mkdir -p "${partial_root}"
    trap 'rm -rf -- "${partial_root}"' EXIT
    echo "${label}: extracting $(basename "${archive}") to ${final_root}." >&2
    if ! (set -o pipefail; zstd -q -d -c "${archive}" | \
      tar --no-same-owner -C "${partial_root}" -xf -); then
      echo "${label}: extraction failed; using ${shared_venv}." >&2
      exit 1
    fi
    if [ ! -x "${partial_root}/bin/python" ]; then
      echo "${label}: extracted interpreter is invalid; using ${shared_venv}." >&2
      exit 1
    fi
    printf '%s\n' "${archive_name}" > "${partial_root}/.node_local_venv_archive"
    mv "${partial_root}" "${final_root}"
    trap - EXIT
    echo "${label}: node-local environment ready." >&2
    printf '%s\n' "${final_root}"
  ) 9>"${lock_file}"
}
