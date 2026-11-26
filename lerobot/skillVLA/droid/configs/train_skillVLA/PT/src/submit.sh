#!/usr/bin/env bash
# Source from a DROID component wrapper; all model/train code stays shared.
droid_stage1_submit() {
  local component="$1" config="$2"; shift 2
  if [[ "$component" != VSA && "$component" != Predictor ]]; then
    echo "Supported DROID components: VSA, Predictor." >&2
    return 2
  fi
  local project configs shared resolver target selector dry=false
  project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
  configs="${project}/lerobot/skillVLA/droid/configs"
  shared="${project}/lerobot/skillVLA/libero/configs"
  export SBD_GLOBAL_CONFIG="${configs}/global_config.yaml"
  export PYTHONPATH="${project}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
  export SBD_GPU_GUARD_TERMINAL_ON_OOM=1
  local -a forwarded=()
  while (( $# )); do
    case "$1" in
      --config) config="${2:?missing YAML}"; shift 2 ;;
      --dry-run) dry=true; shift ;;
      *) forwarded+=("$1"); shift ;;
    esac
  done
  local global_dir
  global_dir="$(dirname "$(realpath "$config")")"
  while [[ "$global_dir" != / && ! -f "$global_dir/global_config.yaml" ]]; do
    global_dir="$(dirname "$global_dir")"
  done
  [[ ! -f "$global_dir/global_config.yaml" ]] || export SBD_GLOBAL_CONFIG="$global_dir/global_config.yaml"
  if [[ "$component" == VSA ]]; then
    resolver="${shared}/train_skillVLA/PT/src/stage1_train_config.py"
    target="${shared}/train_skillVLA/PT/VSA/submit_train.sh"
    selector=STAGE1_TRAIN_CONFIG
    export STAGE1_SUBMIT_DIR="${configs}/train_skillVLA/PT/VSA" STAGE1_JOB_NAME=DROID_SkillVLA_PT
  else
    resolver="${shared}/train_skillVLA/PT/src/auxiliary/auxiliary_train_config.py"
    target="${shared}/train_skillVLA/PT/src/auxiliary/submit_auxiliary.sh"
    selector=SKILL_AUX_TRAIN_CONFIG
    export SKILL_AUX_SUBMIT_DIR="${configs}/train_skillVLA/PT/${component}"
  fi
  # Refuse missing datasets/weights before snapshotting or scheduling GPU work.
  if [[ "$dry" == true ]]; then
    "${project}/.venv/bin/python" "$resolver" --config "$config"
    return
  fi
  "${project}/.venv/bin/python" "$resolver" --config "$config" --shell >/dev/null
  unset TRAIN_DATA STAGE1_ARCHITECTURE_OVERRIDE TERMINATOR_ARCHITECTURE_OVERRIDE
  export "$selector=$config"
  if [[ "$component" == VSA ]]; then
    source "$target" --config "$config" "${forwarded[@]}"
  else
    source "$target" "${forwarded[@]}"
  fi
}
