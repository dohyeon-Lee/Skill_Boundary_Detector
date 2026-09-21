#!/usr/bin/env bash
# Submit NewTask FT for the terminator: warm-start the Stage-1 terminator and continue
# training on the new-task dataset. Reuses the unified auxiliary trainer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKILL_AUX_TRAIN_CONFIG="${SKILL_AUX_TRAIN_CONFIG:-${SCRIPT_DIR}/terminator_ft_config.yaml}"
export SKILL_AUX_SUBMIT_DIR="${SCRIPT_DIR}"
exec "${SCRIPT_DIR}/../../terminator/submit_train.sh" "$@"
