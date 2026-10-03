#!/usr/bin/env bash
# Submit the Stage-1 terminator; it remains a separate training job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

variant="${1:-${TERMINATOR_ARCHITECTURE_OVERRIDE:-}}"
if (( $# > 1 )); then
    echo "Usage: $0 [term1..term15|term16_norm..term20_uni]" >&2
    exit 2
fi
config="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_train_config.yaml}"
export SKILL_AUX_TRAIN_CONFIG="${config}"
export TERMINATOR_ARCHITECTURE_OVERRIDE="${variant}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../terminator/submit_train.sh"
