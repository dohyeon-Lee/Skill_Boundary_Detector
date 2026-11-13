#!/usr/bin/env bash
# Use the shared scheduler implementation with ABC's independent global config.
ABC_CONFIGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABC_PROJECT_ROOT="$(cd "${ABC_CONFIGS_DIR}/../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="${ABC_CONFIGS_DIR}/global_config.yaml"
source "${ABC_PROJECT_ROOT}/lerobot/examples/libero/configs/src/submit_job.sh"

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  _submit_job_cli "$@"
fi
