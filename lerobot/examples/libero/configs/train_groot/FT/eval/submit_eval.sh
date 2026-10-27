#!/usr/bin/env bash
# Run the shared native GR00T evaluator with FT checkpoints and FT-local outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../../src/baseline_eval/submit.sh" groot "${SCRIPT_DIR}" \
  "${GROOT_FT_EVAL_CONFIG:-${SCRIPT_DIR}/groot_eval_config.yaml}" "$@"
