#!/usr/bin/env bash
# Run the shared encoding/report pipeline inside the chosen allocation.
set -euo pipefail
EVAL_DIR="${FSQ_GT_REPLAY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="$(cd "${EVAL_DIR}/../../../../../../.." && pwd)"
export FSQ_GT_REPLAY_DOMAIN=abc
export FSQ_GT_REPLAY_DIR="${EVAL_DIR}"
export FSQ_GT_REPLAY_SRC_DIR="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skills/skill_eval/fsq_codebook/src"
export FSQ_GT_REPLAY_CONFIG="${FSQ_GT_REPLAY_CONFIG:-${EVAL_DIR}/eval_config.yaml}"
exec bash "${FSQ_GT_REPLAY_SRC_DIR}/../submit.sh"
