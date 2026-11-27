#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
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
#   _lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
#   source "${_lib}/src/snapshot_config.sh"
#   CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
snapshot_config() {
  local src="$1"
  local src_dir snap_root bundle snap config_lib global_src servers_src search_dir
  src_dir="$(cd "$(dirname "${src}")" && pwd)"
  src="${src_dir}/$(basename "${src}")"
  snap_root="${src_dir}/.config_snapshots"
  bundle="${snap_root}/$(basename "${src}").$(date +%Y%m%d-%H%M%S)_$$_${RANDOM}"
  snap="${bundle}/$(basename "${src}")"

  # snapshot_config.sh lives in configs/src/, below the repository-wide global_config.yaml.
  # Keep a private copy per submission: a single shared snapshot would still
  # let a later submission mutate an older queued job's global settings.
  if [ -n "${SBD_GLOBAL_CONFIG:-}" ]; then
    global_src="$(cd "$(dirname "${SBD_GLOBAL_CONFIG}")" && pwd)/$(basename "${SBD_GLOBAL_CONFIG}")"
    config_lib="$(dirname "${global_src}")"
  else
    config_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    global_src="${config_lib}/global_config.yaml"
  fi
  if [ ! -f "${global_src}" ]; then
    echo "Global config not found next to snapshot helper: ${global_src}" >&2
    return 1
  fi

  mkdir -p "${bundle}" >&2
  cp "${src}" "${snap}" >&2
  if [ "${src}" != "${global_src}" ]; then
    cp "${global_src}" "${bundle}/global_config.yaml" >&2
  fi
  # Share live profiles across ABC/LIBERO, but freeze a private copy per job.
  # A legacy self-contained snapshot must never inherit today's live profiles.
  servers_src=""
  case "${config_lib}/" in
    */.config_snapshots/*)
      [ ! -d "${config_lib}/servers" ] || servers_src="${config_lib}/servers"
      ;;
    *)
      search_dir="${config_lib}"
      while [ "${search_dir}" != / ]; do
        if [ "$(basename "${search_dir}")" = skillVLA ] && [ -d "${search_dir}/servers" ]; then
          servers_src="${search_dir}/servers"
          break
        fi
        search_dir="$(dirname "${search_dir}")"
      done
      if [ -z "${servers_src}" ] && [ -d "${config_lib}/servers" ]; then
        servers_src="${config_lib}/servers"
      fi
      ;;
  esac
  if [ -n "${servers_src}" ]; then
    cp -r "${servers_src}" "${bundle}/servers" >&2
  fi
  # New Stage-1 component configs inherit shared defaults. Freeze that layer
  # together with the component and global YAML for queued Slurm jobs.
  local stage1_common="${src_dir}/stage1_common_config.yaml"
  [ -f "${stage1_common}" ] || stage1_common="${src_dir}/../stage1_common_config.yaml"
  if [ -f "${stage1_common}" ]; then
    cp "${stage1_common}" "${bundle}/stage1_common_config.yaml" >&2
  fi
  # FT components keep their historical shared-YAML name for snapshot compatibility.
  local newtask_ft_common="${src_dir}/newtask_ft_common_config.yaml"
  [ -f "${newtask_ft_common}" ] || newtask_ft_common="${src_dir}/../newtask_ft_common_config.yaml"
  if [ -f "${newtask_ft_common}" ]; then
    cp "${newtask_ft_common}" "${bundle}/newtask_ft_common_config.yaml" >&2
  fi
  # GR00T FT can derive its PT run from the PT recipe. Resolve it now so a
  # later PT YAML edit cannot change a queued FT job's warm start.
  if [ -f "${bundle}/newtask_ft_common_config.yaml" ] && \
      grep -Eq "^skillvla_backbone:[[:space:]]*[\"']?groot_(integrate|seperate)([\"']|[[:space:]]|$)" "${bundle}/newtask_ft_common_config.yaml"; then
    local backend_helper backend_python
    backend_helper="$(dirname "${BASH_SOURCE[0]}")/../train_skills/src/skillvla_backend.py"
    backend_python="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/.venv/bin/python"
    [ -x "${backend_python}" ] || backend_python=python3
    "${backend_python}" "${backend_helper}" --freeze-ft-source "${snap}" || return
  fi
  # DROID UUID groups are data contracts too: queued work must not reread a
  # subsequently edited allowlist. Other domains and old snapshots are unchanged.
  local group_helper group_python
  group_helper="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/droid/configs/src/dataset_groups.py"
  if [ -f "${group_helper}" ] && grep -Eq '^workflow_domain: *droid *$' "${bundle}/global_config.yaml"; then
    group_python="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/.venv/bin/python"
    [ -x "${group_python}" ] || group_python=python3
    "${group_python}" "${group_helper}" --freeze "${bundle}/global_config.yaml" || return
  fi
  printf '%s\n' "${snap}"
}
