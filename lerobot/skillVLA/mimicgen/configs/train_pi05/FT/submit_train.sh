#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/pi05_FT_config.yaml"
SELECTOR=TRAIN_PI05_CONFIG PREFIX=FT BATCH_SCRIPT=train.sbatch
source "${HERE}/../src/launch.sh" "$@"
