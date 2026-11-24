#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/pi05_eval_config.yaml"
SELECTOR=PI05_EVAL_CONFIG PREFIX=EVAL BATCH_SCRIPT=eval.sbatch
source "${HERE}/../../src/launch.sh" "$@"
