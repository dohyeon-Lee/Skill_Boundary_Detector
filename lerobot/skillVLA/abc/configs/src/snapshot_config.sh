#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../remote.sh"
fi
# Reuse the shared snapshot implementation, selecting ABC's config and servers.
ABC_CONFIGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABC_PROJECT_ROOT="$(cd "${ABC_CONFIGS_DIR}/../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="${ABC_CONFIGS_DIR}/global_config.yaml"
source "${ABC_PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src/snapshot_config.sh"
