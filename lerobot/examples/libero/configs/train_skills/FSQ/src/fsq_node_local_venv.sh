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
  ) 9>"${lock_file}"
  printf '%s\n' "${archive}"
}


fsq_stage_venv_on_node() {
  local archive="${1:?fsq_stage_venv_on_node needs an archive}"
  local shared_venv="${2:?fsq_stage_venv_on_node needs a fallback venv}"
  if [ ! -s "${archive}" ] || [ -z "${SLURM_TMPDIR:-}" ]; then
    echo "FSQ venv cache: archive or SLURM_TMPDIR unavailable; using ${shared_venv}." >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1; then
    echo "FSQ venv cache: zstd unavailable on compute node; using ${shared_venv}." >&2
    return 1
  fi

  local local_root="${SLURM_TMPDIR}/fsq_venv_${SLURM_JOB_ID:-$$}"
  local local_archive="${SLURM_TMPDIR}/$(basename "${archive}")"
  mkdir -p "${local_root}"
  echo "FSQ venv cache: copying one archive to ${SLURM_TMPDIR}." >&2
  if ! cp "${archive}" "${local_archive}"; then
    echo "FSQ venv cache: copy failed; using ${shared_venv}." >&2
    return 1
  fi
  if ! zstd -q -d -c "${local_archive}" | tar --no-same-owner -C "${local_root}" -xf -; then
    echo "FSQ venv cache: extraction failed; using ${shared_venv}." >&2
    return 1
  fi
  if [ ! -x "${local_root}/bin/python" ]; then
    echo "FSQ venv cache: extracted interpreter is invalid; using ${shared_venv}." >&2
    return 1
  fi
  printf '%s\n' "${local_root}"
}
