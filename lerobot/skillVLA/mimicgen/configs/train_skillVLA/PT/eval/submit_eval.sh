#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../../.." && pwd)"
PROJECT="$(cd "${CONFIGS}/../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="${CONFIGS}/global_config.yaml"
export STAGE1_EVAL_CONFIG="${HERE}/pt_eval_config.yaml"
if [[ "${1:-}" == --config ]]; then export STAGE1_EVAL_CONFIG="${2:?missing YAML}"; shift 2; fi
if [[ "${1:-}" == --dry-run ]]; then export STAGE1_EVAL_DRY_RUN=1; shift; fi
export STAGE1_EVAL_ENV_TYPE=mimicgen STAGE1_EVAL_WORK_DIR="${HERE}"
export STAGE1_EVAL_CONFIG_RESOLVER="${CONFIGS}/train_skillVLA/PT/eval/src/config.py"
export PYTHONPATH="${PROJECT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
source "${PROJECT}/lerobot/skillVLA/libero/configs/train_skillVLA/PT/eval/submit_eval.sh" "$@"
