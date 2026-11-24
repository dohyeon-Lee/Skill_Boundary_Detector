#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}"
while [[ ! -f "${ROOT}/setup_env.sh" ]]; do ROOT="$(dirname "${ROOT}")"; done
export SBD_GLOBAL_CONFIG="${ROOT}/lerobot/skillVLA/mimicgen/configs/global_config.yaml"
source "${ROOT}/lerobot/skillVLA/libero/configs/src/submit_args.sh"
source "${ROOT}/lerobot/skillVLA/libero/configs/src/submit_job.sh"
export PYTHONPATH="${ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
EXPORTS="$("${ROOT}/.venv/bin/python" "${HERE}/src/skill_start_probe.py" --freeze "${HERE}/ablation_config.yaml")"
eval "${EXPORTS}"
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
[[ "$#" == 0 ]] || { echo "Usage: ./submit_eval.sh [--JOBID] [--dry-run]" >&2; exit 2; }
submit_job --job-name=Coffee_skill_start_probe --gres="${PROBE_GRES}" \
  --cpus-per-task="${PROBE_CPUS}" --mem="${PROBE_MEMORY}" --time="${PROBE_TIME}" \
  --output="${HERE}/logs/%x_%j.out" --error="${HERE}/logs/%x_%j.err" \
  "${HERE}/src/run_probe.sbatch"
