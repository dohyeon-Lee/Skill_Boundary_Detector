#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../.." && pwd)"
source "${CONFIGS}/src/workflow.sh"
mimicgen_workflow sbd "${HERE}/build_data_config.yaml" TRAIN_SKILLS_CONFIG train_skills/build_data/submit_build_data.sh "$@"
