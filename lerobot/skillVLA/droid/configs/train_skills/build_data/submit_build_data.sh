#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"; fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="${PROJECT_ROOT}/lerobot/skillVLA/droid/configs/global_config.yaml"
export TRAIN_SKILLS_CONFIG="${HERE}/build_data_config.yaml"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
unset TRAIN_DATA
plan=false
forwarded=()
while (( $# )); do
  case "$1" in
    --config) export TRAIN_SKILLS_CONFIG="${2:?missing YAML}"; shift 2 ;;
    --dry-run) plan=true; shift ;;
    *) forwarded+=("$1"); shift ;;
  esac
done
if [[ "${plan}" == true ]]; then
  exec "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skills/src/train_skills_config.py" --config "${TRAIN_SKILLS_CONFIG}" --stage build_data
fi
source "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skills/build_data/submit_build_data.sh" "${forwarded[@]}"
