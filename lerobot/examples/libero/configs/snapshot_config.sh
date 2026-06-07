#!/usr/bin/env bash
# Shared helper: freeze a launcher's YAML config at SUBMIT time.
#
# Slurm only spools the .sbatch script text; every launcher re-reads its YAML (re-runs the config
# emitter) on the compute node when the job actually starts. So editing the repo YAML while a job
# is still queued would silently change that pending job. snapshot_config copies the YAML to an
# immutable per-submit snapshot and prints its path; the submit script then points the job's
# *_CONFIG env var at that snapshot, so each job keeps exactly the config it was submitted with.
# Snapshots live in .config_snapshots/ next to the YAML (kept for provenance).
#
# Usage (in a submit_*.sh, after resolving CONFIG_PATH, before sbatch):
#   _lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
#   source "${_lib}/snapshot_config.sh"
#   CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
snapshot_config() {
  local src="$1"
  local snap_dir
  snap_dir="$(dirname "${src}")/.config_snapshots"
  mkdir -p "${snap_dir}" >&2
  local snap="${snap_dir}/$(basename "${src}").$(date +%Y%m%d-%H%M%S)_$$_${RANDOM}"
  cp "${src}" "${snap}" >&2
  printf '%s\n' "${snap}"
}
