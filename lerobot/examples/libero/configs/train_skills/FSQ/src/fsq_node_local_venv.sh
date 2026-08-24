#!/usr/bin/env bash
# FSQ-only helpers for turning the many-file project venv into one sequential
# Lustre read, then running from node-local storage. Other training stages do
# not source this file and the shared venv is never modified.

fsq_prepare_venv_archive() {
  local project_root="${1:?fsq_prepare_venv_archive needs PROJECT_ROOT}"
  local source_venv
  source_venv="$(readlink -f "${project_root}/.venv")"
  if [ ! -x "${source_venv}/bin/python" ]; then
    echo "FSQ venv cache: shared interpreter is missing: ${source_venv}/bin/python" >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "FSQ venv cache: zstd/flock unavailable; using the shared venv." >&2
    return 1
  fi

  local archive_root="${FSQ_VENV_ARCHIVE_DIR:-${project_root}/.cache/fsq_venv}"
  mkdir -p "${archive_root}"

  # Dependency lock + interpreter metadata define the cache generation. A uv
  # lock change creates a new immutable archive without touching older jobs.
  local fingerprint
  fingerprint="$({
    printf '%s\n' "${source_venv}"
    sha256sum "${project_root}/uv.lock" "${project_root}/pyproject.toml"
    sha256sum "${source_venv}/pyvenv.cfg"
  } | sha256sum | cut -d' ' -f1)"

  local archive="${archive_root}/venv-${fingerprint:0:16}.tar.zst"
  local size_file="${archive}.size"
  local lock_file="${archive}.lock"
  (
    flock 9
    if [ ! -s "${archive}" ]; then
      local temporary="${archive}.tmp.$$"
      trap 'rm -f "${temporary}"' EXIT
      echo "FSQ venv cache: creating one-time archive ${archive}" >&2
      tar -C "${source_venv}" -cf - . | zstd -1 -T"${FSQ_VENV_ARCHIVE_THREADS:-4}" -q -o "${temporary}"
      mv "${temporary}" "${archive}"
      trap - EXIT
      echo "FSQ venv cache: archive ready." >&2
    fi
    if [ ! -s "${size_file}" ] \
      || ! [[ "$(tr -dc '0-9' < "${size_file}")" =~ ^[0-9]+$ ]]; then
      local size_tmp="${size_file}.tmp.$$"
      du -s --apparent-size --block-size=1 "${source_venv}" | awk '{print $1}' > "${size_tmp}"
      mv "${size_tmp}" "${size_file}"
    fi
  ) 9>"${lock_file}"
  printf '%s\n' "${archive}"
}


fsq_stage_venv_on_node() {
  local archive="${1:?fsq_stage_venv_on_node needs an archive}"
  local shared_venv="${2:?fsq_stage_venv_on_node needs a fallback venv}"
  if [ ! -s "${archive}" ]; then
    echo "FSQ venv cache: archive unavailable; using ${shared_venv}." >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
    echo "FSQ venv cache: zstd/flock unavailable on compute node; using ${shared_venv}." >&2
    return 1
  fi

  local owner="${USER:-}"
  if [ -z "${owner}" ]; then
    owner="$(id -un)"
  fi
  local local_root="${FSQ_VENV_LOCAL_ROOT:-}"
  if [ -z "${local_root}" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    local_root="${SLURM_TMPDIR}/fsq_venv"
  fi
  if [ -z "${local_root}" ] && [ -d /dev/shm ] && [ -w /dev/shm ]; then
    local_root="/dev/shm/${owner}/fsq_venv"
  fi
  if [ -z "${local_root}" ] && [ -d /tmp ] && [ -w /tmp ]; then
    local_root="/tmp/${owner}/fsq_venv"
  fi
  if [ -z "${local_root}" ] || [[ "${local_root}" != /* ]]; then
    echo "FSQ venv cache: no absolute writable local root; using ${shared_venv}." >&2
    return 1
  fi
  if ! mkdir -p "${local_root}"; then
    echo "FSQ venv cache: cannot create ${local_root}; using ${shared_venv}." >&2
    return 1
  fi

  local archive_name fingerprint
  archive_name="$(basename "${archive}")"
  fingerprint="${archive_name%.tar.zst}"
  case "${fingerprint}" in
    *[!A-Za-z0-9._-]*)
      echo "FSQ venv cache: unsafe archive name ${archive_name}; using ${shared_venv}." >&2
      return 1
      ;;
  esac

  local final_root="${local_root}/${fingerprint}"
  local lock_file="${local_root}/.${fingerprint}.lock"
  (
    flock -x 9
    if [ -x "${final_root}/bin/python" ] \
      && [ -f "${final_root}/.fsq_venv_archive" ] \
      && [ "$(cat "${final_root}/.fsq_venv_archive")" = "${archive_name}" ]; then
      echo "FSQ venv cache: reusing ${final_root}." >&2
      printf '%s\n' "${final_root}"
      exit 0
    fi

    local expected_bytes
    if [ -s "${archive}.size" ]; then
      expected_bytes="$(tr -dc '0-9' < "${archive}.size")"
    else
      expected_bytes=$(( $(stat -c %s "${archive}") * 3 ))
    fi
    if [ -z "${expected_bytes}" ]; then
      echo "FSQ venv cache: invalid size metadata; using ${shared_venv}." >&2
      exit 1
    fi
    local available_bytes reserve_bytes
    available_bytes="$(df -P --block-size=1 "${local_root}" | awk 'NR == 2 {print $4}')"
    reserve_bytes=$((8 * 1024 * 1024 * 1024))
    if (( expected_bytes > available_bytes \
      || available_bytes - expected_bytes < reserve_bytes )); then
      echo "FSQ venv cache: insufficient local space at ${local_root} " \
        "(available=${available_bytes}, payload=${expected_bytes}, reserve=${reserve_bytes}); " \
        "using ${shared_venv}." >&2
      exit 1
    fi

    if [ -e "${final_root}" ]; then
      local invalid_root="${local_root}/.invalid"
      mkdir -p "${invalid_root}"
      mv "${final_root}" "${invalid_root}/${fingerprint}.$(date +%s).$$"
    fi
    local partial_root="${local_root}/.${fingerprint}.partial.$$"
    mkdir -p "${partial_root}"
    trap 'rm -rf -- "${partial_root}"' EXIT
    echo "FSQ venv cache: extracting one sequential archive to ${final_root}." >&2
    if ! zstd -q -d -c "${archive}" \
      | tar --no-same-owner -C "${partial_root}" -xf -; then
      echo "FSQ venv cache: extraction failed; using ${shared_venv}." >&2
      exit 1
    fi
    if [ ! -x "${partial_root}/bin/python" ]; then
      echo "FSQ venv cache: extracted interpreter is invalid; using ${shared_venv}." >&2
      exit 1
    fi
    printf '%s\n' "${archive_name}" > "${partial_root}/.fsq_venv_archive"
    mv "${partial_root}" "${final_root}"
    trap - EXIT
    echo "FSQ venv cache: local venv ready at ${final_root}." >&2
    printf '%s\n' "${final_root}"
  ) 9>"${lock_file}"
  local status=$?
  if [ "${status}" -ne 0 ]; then
    return 1
  fi
}
