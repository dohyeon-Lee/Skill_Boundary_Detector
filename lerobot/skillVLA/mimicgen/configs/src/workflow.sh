#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
# Domain selectors must never inherit an unrelated workflow's snapshot.
mimicgen_workflow() {
  local kind="$1" config="$2" selector="$3" target="$4"
  shift 4
  export SBD_GLOBAL_CONFIG="${CONFIGS}/global_config.yaml"
  local project shared
  project="$(cd "${CONFIGS}/../../../.." && pwd)"
  shared="${project}/lerobot/skillVLA/libero/configs"
  local plan=false
  local -a forwarded=()
  while (( $# )); do
    case "$1" in
      --config) config="${2:?missing YAML}"; shift 2 ;;
      --dry-run) plan=true; shift ;;
      *) forwarded+=("$1"); shift ;;
    esac
  done
  unset TRAIN_DATA STAGE1_ARCHITECTURE_OVERRIDE TERMINATOR_ARCHITECTURE_OVERRIDE
  export "$selector=$config"
  if [[ "${plan}" == true ]]; then
    PYTHONPATH="${project}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}" "${project}/.venv/bin/python" "${CONFIGS}/src/workflow.py" "$kind" --config "$config"
    return
  fi
  # Source skips the shared launcher's direct-execution remote hook. Its worker,
  # resolver and model code stay shared while the explicit YAML owns the domain.
  source "${shared}/${target}" "${forwarded[@]}"
}
