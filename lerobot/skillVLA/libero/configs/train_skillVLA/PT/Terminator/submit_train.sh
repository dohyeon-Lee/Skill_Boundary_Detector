#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Submit the Stage-1 terminator; it remains a separate training job.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

variant="${1:-${TERMINATOR_ARCHITECTURE_OVERRIDE:-}}"
if (( $# > 1 )); then
    echo "Usage: $0 [registered Terminator architecture, e.g. Term_Token_1_dinoS_freeze]" >&2
    exit 2
fi
config="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_train_config.yaml}"
export SKILL_AUX_TRAIN_CONFIG="${config}"
export TERMINATOR_ARCHITECTURE_OVERRIDE="${variant}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/src/submit_auxiliary.sh"
