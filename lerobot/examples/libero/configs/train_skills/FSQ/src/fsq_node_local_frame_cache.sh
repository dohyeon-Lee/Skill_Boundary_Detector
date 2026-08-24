#!/usr/bin/env bash
# FSQ/BSQ-only helper for staging the immutable RGB cache from shared Lustre to
# node-local storage. Multiple jobs on one node share a fingerprinted copy.

_fsq_local_frame_cache_matches() {
  local shared_cache="${1:?shared cache is required}"
  local local_cache="${2:?local cache is required}"
  [ -f "${local_cache}/manifest.json" ] \
    && [ -f "${local_cache}/_SUCCESS" ] \
    && cmp -s "${shared_cache}/manifest.json" "${local_cache}/manifest.json" \
    && cmp -s "${shared_cache}/_SUCCESS" "${local_cache}/_SUCCESS"
}


_fsq_auto_frame_cache_root() {
  local owner="${USER:-}"
  if [ -z "${owner}" ]; then
    owner="$(id -un)"
  fi
  if [ -d /dev/shm ] && [ -w /dev/shm ]; then
    printf '%s\n' "/dev/shm/${owner}/fsq_frame_cache"
    return 0
  fi
  if [ -d /tmp ] && [ -w /tmp ]; then
    printf '%s\n' "/tmp/${owner}/fsq_frame_cache"
    return 0
  fi
  return 1
}


fsq_stage_frame_cache_on_node() {
  local shared_cache="${1:?fsq_stage_frame_cache_on_node needs a shared cache}"
  local configured_root="${2:-}"
  local reserve_gb="${3:-16}"

  if [ ! -f "${shared_cache}/manifest.json" ] || [ ! -f "${shared_cache}/_SUCCESS" ]; then
    echo "FSQ frame cache stage: shared cache is incomplete; using ${shared_cache}." >&2
    return 1
  fi
  if ! command -v flock >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
    echo "FSQ frame cache stage: flock/rsync unavailable; using ${shared_cache}." >&2
    return 1
  fi
  if ! [[ "${reserve_gb}" =~ ^[0-9]+$ ]]; then
    echo "FSQ frame cache stage: reserve_gb must be a non-negative integer." >&2
    return 1
  fi

  local fingerprint
  fingerprint="$(tr -d '\r\n' < "${shared_cache}/_SUCCESS")"
  if [ -z "${fingerprint}" ] || [ "${fingerprint}" != "$(basename "${shared_cache}")" ]; then
    echo "FSQ frame cache stage: completion marker/fingerprint mismatch in ${shared_cache}." >&2
    return 1
  fi
  case "${fingerprint}" in
    *[!A-Za-z0-9._-]*)
      echo "FSQ frame cache stage: unsafe fingerprint ${fingerprint}." >&2
      return 1
      ;;
  esac

  local local_root="${configured_root}"
  if [ -z "${local_root}" ]; then
    if ! local_root="$(_fsq_auto_frame_cache_root)"; then
      echo "FSQ frame cache stage: no writable node-local root; using ${shared_cache}." >&2
      return 1
    fi
  fi
  if [[ "${local_root}" != /* ]]; then
    echo "FSQ frame cache stage: local root must be absolute; using ${shared_cache}." >&2
    return 1
  fi
  if ! mkdir -p "${local_root}"; then
    echo "FSQ frame cache stage: cannot create ${local_root}; using ${shared_cache}." >&2
    return 1
  fi

  local final_cache="${local_root}/${fingerprint}"
  local partial_cache="${local_root}/.${fingerprint}.partial"
  local lock_file="${local_root}/.${fingerprint}.lock"
  (
    flock -x 9
    if _fsq_local_frame_cache_matches "${shared_cache}" "${final_cache}"; then
      echo "FSQ frame cache stage: reusing ${final_cache}." >&2
      printf '%s\n' "${final_cache}"
      exit 0
    fi

    if [ -e "${final_cache}" ]; then
      local invalid_root="${local_root}/.invalid"
      local invalid_cache
      mkdir -p "${invalid_root}"
      invalid_cache="${invalid_root}/${fingerprint}.$(date +%s).$$"
      if ! mv "${final_cache}" "${invalid_cache}"; then
        echo "FSQ frame cache stage: cannot quarantine invalid ${final_cache}." >&2
        exit 1
      fi
      echo "FSQ frame cache stage: quarantined invalid copy at ${invalid_cache}." >&2
    fi
    mkdir -p "${partial_cache}"

    local source_bytes partial_bytes available_bytes reserve_bytes required_bytes
    # The v2 payload is already lossless-zstd compressed (~68 GiB for libero_90),
    # so its apparent size is also the amount tmpfs must hold. Apparent size
    # remains the conservative check and is also correct for legacy v1 input.
    source_bytes="$(du -s --apparent-size --block-size=1 "${shared_cache}" | awk '{print $1}')"
    partial_bytes="$(du -s --block-size=1 "${partial_cache}" | awk '{print $1}')"
    available_bytes="$(df -P --block-size=1 "${local_root}" | awk 'NR == 2 {print $4}')"
    reserve_bytes=$((reserve_gb * 1024 * 1024 * 1024))
    required_bytes=$((source_bytes - partial_bytes + reserve_bytes))
    if (( required_bytes < reserve_bytes )); then
      required_bytes=${reserve_bytes}
    fi
    if (( available_bytes < required_bytes )); then
      echo "FSQ frame cache stage: insufficient local space at ${local_root} " \
        "(available=${available_bytes}, required=${required_bytes}); using shared cache." >&2
      exit 1
    fi
    if [ -n "${SLURM_MEM_PER_NODE:-}" ] && [[ "${SLURM_MEM_PER_NODE}" =~ ^[0-9]+$ ]]; then
      local slurm_memory_bytes=$((SLURM_MEM_PER_NODE * 1024 * 1024))
      if (( slurm_memory_bytes < source_bytes + reserve_bytes )); then
        echo "FSQ frame cache stage: Slurm memory allocation is too small " \
          "(allocated=${slurm_memory_bytes}, required=$((source_bytes + reserve_bytes))); " \
          "using shared cache." >&2
        exit 1
      fi
    fi

    echo "FSQ frame cache stage: copying ${shared_cache} -> ${final_cache} " \
      "(logical bytes=${source_bytes})." >&2
    # The unpublished partial tree retains completed files after interruption,
    # so a later job can resume safely.
    if ! rsync -aS --delete "${shared_cache}/" "${partial_cache}/"; then
      echo "FSQ frame cache stage: copy failed; using ${shared_cache}." >&2
      exit 1
    fi
    if ! _fsq_local_frame_cache_matches "${shared_cache}" "${partial_cache}"; then
      echo "FSQ frame cache stage: copied marker/manifest validation failed." >&2
      exit 1
    fi
    if ! mv "${partial_cache}" "${final_cache}"; then
      echo "FSQ frame cache stage: atomic publish failed; using ${shared_cache}." >&2
      exit 1
    fi
    echo "FSQ frame cache stage: local cache ready at ${final_cache}." >&2
    printf '%s\n' "${final_cache}"
  ) 9>"${lock_file}"
}
