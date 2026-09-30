#!/usr/bin/env bash
# Submit the Stage-1 terminator; it remains a separate training job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

variant="${1:-}"
if (( $# > 1 )); then
    echo "Usage: $0 [term6|term7|term8|term9]" >&2
    exit 2
fi

case "${variant}" in
    "") config="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_train_config.yaml}" ;;
    term6|hist20_top_goalxyz)
        config="${SCRIPT_DIR}/term6_hist20_top_goalxyz.yaml" ;;
    term7|hist20_top_goalxyz_progress_detached)
        config="${SCRIPT_DIR}/term7_hist20_top_goalxyz_progress_detached.yaml" ;;
    term8|hist20_top_nogoal)
        config="${SCRIPT_DIR}/term8_hist20_top_nogoal.yaml" ;;
    term9|hist20_both_goalxyz)
        config="${SCRIPT_DIR}/term9_hist20_both_goalxyz.yaml" ;;
    *)
        echo "Unknown variant '${variant}'." >&2
        echo "Usage: $0 [term6|term7|term8|term9]" >&2
        exit 2
        ;;
esac

export SKILL_AUX_TRAIN_CONFIG="${config}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../terminator/submit_train.sh"
