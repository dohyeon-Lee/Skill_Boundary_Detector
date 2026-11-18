#!/usr/bin/env bash
# Run the shared native MolmoAct2 evaluator with FT checkpoints and FT-local outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../../src/baseline_eval/submit.sh" molmoact2 "${SCRIPT_DIR}" \
  "${MOLMOACT2_FT_EVAL_CONFIG:-${SCRIPT_DIR}/molmoact2_eval_config.yaml}" "$@"
