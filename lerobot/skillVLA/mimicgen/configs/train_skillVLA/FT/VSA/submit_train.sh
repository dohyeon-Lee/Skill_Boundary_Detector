#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../../.." && pwd)"
source "${CONFIGS}/src/workflow.sh"
mimicgen_workflow ft_vsa "${HERE}/vsa_ft_config.yaml" NEWTASK_FT_VSA_CONFIG train_skillVLA/FT/VSA/submit_train.sh "$@"
