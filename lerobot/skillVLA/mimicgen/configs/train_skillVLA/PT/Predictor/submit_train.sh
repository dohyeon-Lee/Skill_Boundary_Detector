#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../../.." && pwd)"
source "${CONFIGS}/src/workflow.sh"
mimicgen_workflow pt_predictor "${HERE}/predictor_train_config.yaml" SKILL_AUX_TRAIN_CONFIG train_skillVLA/PT/Predictor/submit_train.sh "$@"
