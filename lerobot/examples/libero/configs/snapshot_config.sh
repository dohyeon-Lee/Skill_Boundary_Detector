#!/usr/bin/env bash
# Shared helper: freeze a launcher's local and global YAML config at SUBMIT time.
#
# Slurm only spools the .sbatch script text; every launcher re-reads its YAML (re-runs the config
# emitter) on the compute node when the job actually starts. So editing the repo YAML while a job
# is still queued would silently change that pending job. snapshot_config copies the YAML to an
# immutable per-submit bundle together with the shared global_config.yaml and prints the local
# snapshot path. The config loader finds the bundled global_config.yaml first, so both layers stay
# exactly as they were at submission time. Bundles live in .config_snapshots/ next to the local
# YAML (kept for provenance).
#
# Usage (in a submit_*.sh, after resolving CONFIG_PATH, before sbatch):
#   _lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
#   source "${_lib}/snapshot_config.sh"
#   CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
snapshot_config() {
  local src="$1"
  local src_dir snap_root bundle snap config_lib global_src
  src_dir="$(cd "$(dirname "${src}")" && pwd)"
  src="${src_dir}/$(basename "${src}")"
  snap_root="${src_dir}/.config_snapshots"
  bundle="${snap_root}/$(basename "${src}").$(date +%Y%m%d-%H%M%S)_$$_${RANDOM}"
  snap="${bundle}/$(basename "${src}")"

  # snapshot_config.sh lives next to the repository-wide global_config.yaml.
  # Keep a private copy per submission: a single shared snapshot would still
  # let a later submission mutate an older queued job's global settings.
  config_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  global_src="${config_lib}/global_config.yaml"
  if [ ! -f "${global_src}" ]; then
    echo "Global config not found next to snapshot helper: ${global_src}" >&2
    return 1
  fi

  mkdir -p "${bundle}" >&2
  cp "${src}" "${snap}" >&2
  if [ "${src}" != "${global_src}" ]; then
    cp "${global_src}" "${bundle}/global_config.yaml" >&2
  fi
  printf '%s\n' "${snap}"
}
