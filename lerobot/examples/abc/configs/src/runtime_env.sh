#!/usr/bin/env bash
# Runtime cache and credential handling is common to ABC and LIBERO.
ABC_CONFIGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABC_PROJECT_ROOT="$(cd "${ABC_CONFIGS_DIR}/../../../.." && pwd)"
source "${ABC_PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
