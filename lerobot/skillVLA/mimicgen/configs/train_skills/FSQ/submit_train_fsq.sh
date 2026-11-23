#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../.." && pwd)"
source "${CONFIGS}/src/workflow.sh"
mimicgen_workflow fsq "${HERE}/fsq_config.yaml" TRAIN_SKILLS_CONFIG train_skills/FSQ/submit_train_fsq.sh "$@"
