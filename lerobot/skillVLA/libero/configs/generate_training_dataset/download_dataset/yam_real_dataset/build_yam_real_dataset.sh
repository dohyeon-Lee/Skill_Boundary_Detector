#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${YAM_CONFIG:-${SCRIPT_DIR}/yam_real_dataset_config.yaml}"

eval "$(python "${SCRIPT_DIR}/src/yam_real_dataset_config.py" --config "${CONFIG}" --shell)"

SETS="${YAM_ONLY:-${YAM_SET_NAMES}}"
if [[ -z "${SETS// }" ]]; then
  echo "No YAM sets selected" >&2
  exit 1
fi

for set_name in ${SETS}; do
  args=(--config "${CONFIG}" --set "${set_name}")
  if [[ "${FORCE:-0}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ -n "${MAX_EPISODES:-}" ]]; then
    args+=(--max-episodes "${MAX_EPISODES}")
  fi
  python "${SCRIPT_DIR}/src/convert_yam_lerobot.py" "${args[@]}"
done
