#!/usr/bin/env bash
# ABC entry point for the shared FSQ codebook/GT-frame evaluator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
SHARED_DIR="${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/fsq_codebook"

export FSQ_GT_REPLAY_DOMAIN=abc
export FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}"
export FSQ_GT_REPLAY_SRC_DIR="${SHARED_DIR}/src"
export FSQ_GT_REPLAY_CONFIG="${FSQ_GT_REPLAY_CONFIG:-${SCRIPT_DIR}/config.yaml}"

exec bash "${SHARED_DIR}/submit.sh"
