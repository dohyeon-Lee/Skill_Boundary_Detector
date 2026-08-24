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
  ) 9>"${lock_file}"; then
    return 1
  fi
  printf '%s\n' "${archive}"
}


stage_node_local_venv() {
  local archive="${1:?stage_node_local_venv needs an archive}"
  local shared_venv="${2:?stage_node_local_venv needs a fallback venv}"
  local label="${3:-Node-local venv}"
  if [ ! -s "${archive}" ] || [ -z "${SLURM_TMPDIR:-}" ] || [ ! -d "${SLURM_TMPDIR:-}" ]; then
    echo "${label}: archive or SLURM_TMPDIR unavailable; using ${shared_venv}." >&2
    return 1
  fi
  if ! command -v zstd >/dev/null 2>&1; then
    echo "${label}: zstd unavailable on compute node; using ${shared_venv}." >&2
    return 1
  fi

  local job_tag="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_RESTART_COUNT:-0}_$$"
  job_tag="${job_tag//[^A-Za-z0-9_.-]/_}"
  local local_root="${SLURM_TMPDIR}/node_local_venv_${job_tag}"
  local local_archive="${SLURM_TMPDIR}/venv_${job_tag}.tar.zst"
  mkdir -p "${local_root}"
  echo "${label}: copying $(basename "${archive}") to ${SLURM_TMPDIR}." >&2
  if ! cp "${archive}" "${local_archive}"; then
    rm -f -- "${local_archive}"
    echo "${label}: copy failed; using ${shared_venv}." >&2
    return 1
  fi
  echo "${label}: extracting the node-local Python environment." >&2
  if ! (set -o pipefail; zstd -q -d -c "${local_archive}" | \
    tar --no-same-owner -C "${local_root}" -xf -); then
    rm -f -- "${local_archive}"
    echo "${label}: extraction failed; using ${shared_venv}." >&2
    return 1
  fi
  rm -f -- "${local_archive}"
  if [ ! -x "${local_root}/bin/python" ]; then
    echo "${label}: extracted interpreter is invalid; using ${shared_venv}." >&2
    return 1
  fi
  printf '%s\n' "${local_root}"
}
