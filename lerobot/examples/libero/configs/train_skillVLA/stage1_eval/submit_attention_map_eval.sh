#!/usr/bin/env bash
# Dedicated attention-map entry point; Stage-1 eval handles rollout/loading.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE1_EVAL_CONFIG="${1:-${SCRIPT_DIR}/attention_map_eval_config.yaml}"
exec "${SCRIPT_DIR}/submit_eval.sh"
