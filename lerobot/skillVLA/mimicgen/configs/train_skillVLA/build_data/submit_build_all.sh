#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="$(cd "${HERE}/../.." && pwd)"
PROJECT="$(cd "${CONFIGS}/../../../.." && pwd)"
export SBD_GLOBAL_CONFIG="${CONFIGS}/global_config.yaml"
source "${CONFIGS}/src/submit_args.sh"
CONFIG="${HERE}/build_data_config.yaml"
if [[ "${1:-}" == --config ]]; then CONFIG="${2:?missing YAML}"; shift 2; fi
export PYTHONPATH="${PROJECT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${1:-}" == --dry-run ]]; then
  exec "${PROJECT}/.venv/bin/python" "${HERE}/src/build_dataset.py" --config "${CONFIG}" --dry-run
fi
[[ $# == 0 ]] || { echo "Usage: $0 [--JOBID] [--config FILE] [--dry-run]" >&2; exit 2; }
source "${CONFIGS}/src/snapshot_config.sh"
CONFIG="$(snapshot_config "${CONFIG}")"
"${PROJECT}/.venv/bin/python" "${HERE}/src/build_dataset.py" --config "${CONFIG}" --freeze-sbd
export MIMICGEN_BUILD_CONFIG="${CONFIG}" MIMICGEN_BUILD_DIR="${HERE}"
source "${CONFIGS}/src/submit_job.sh"
mapfile -t RESOURCES < <("${PROJECT}/.venv/bin/python" - "${CONFIG}" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))['slurm']
for key in ['gres','cpus','memory','time']: print(cfg[key])
PY
)
mkdir -p "${HERE}/logs"
cd "${HERE}"
submit_job --job-name=MimicGen_SkillVLA_build --gres="${RESOURCES[0]}" --cpus-per-task="${RESOURCES[1]}" --mem="${RESOURCES[2]}" --time="${RESOURCES[3]}" --output=logs/%x_%j.out --error=logs/%x_%j.err "${HERE}/src/build.sbatch"
