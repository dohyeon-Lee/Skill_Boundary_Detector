#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
export SKILL_AUX_TRAIN_CONFIG="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/predictor_train_config.yaml}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
if (( $# )); then echo "Usage: $0 [--JOB_ID]" >&2; exit 2; fi
exec "${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skillVLA/PT/Terminator/src/submit_auxiliary.sh"
