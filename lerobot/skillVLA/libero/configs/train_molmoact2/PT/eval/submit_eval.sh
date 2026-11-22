#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../../src/baseline_eval/submit.sh" molmoact2 "${SCRIPT_DIR}" \
  "${MOLMOACT2_EVAL_CONFIG:-${SCRIPT_DIR}/molmoact2_eval_config.yaml}" "$@"
