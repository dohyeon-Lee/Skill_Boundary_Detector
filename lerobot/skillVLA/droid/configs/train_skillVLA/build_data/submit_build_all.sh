#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
SHARED="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src"
export SBD_GLOBAL_CONFIG="${SCRIPT_DIR}/../../global_config.yaml"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
source "${SHARED}/submit_args.sh"
CONFIG="${SCRIPT_DIR}/train_skillVLA_config.yaml"
DRY=false
while (( $# )); do
  case "$1" in
    --config) CONFIG="${2:?missing YAML}"; shift 2 ;;
    --dry-run) DRY=true; shift ;;
    *) echo 'Usage: submit_build_all.sh [--JOBID|--yonsei] [--config YAML] [--dry-run]' >&2; exit 2 ;;
  esac
done
PLAN="$("${PROJECT_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/src/build_dataset.py" --config "$CONFIG")"
if [[ "$DRY" == true ]]; then printf '%s\n' "$PLAN"; exit; fi
source "${SHARED}/snapshot_config.sh"
CONFIG="$(snapshot_config "$CONFIG")"
source "${SHARED}/submit_job.sh"
export DROID_SKILLVLA_BUILD_CONFIG="$CONFIG" DROID_SKILLVLA_BUILD_DIR="$SCRIPT_DIR"
mapfile -t VALUES < <("${PROJECT_ROOT}/.venv/bin/python" -c 'import json,sys;p=json.loads(sys.argv[1]);print("\n".join(str(p[k]) for k in ("cpus","memory","time")))' "$PLAN")
mkdir -p "${SCRIPT_DIR}/logs"
cd "$SCRIPT_DIR"
submit_job --job-name=DROID_SkillVLA_build --gres=gpu:1 --cpus-per-task="${VALUES[0]}" --mem="${VALUES[1]}" --time="${VALUES[2]}" \
  --output="${SCRIPT_DIR}/logs/%x_%j.out" --error="${SCRIPT_DIR}/logs/%x_%j.err" "${SCRIPT_DIR}/src/build.sbatch"
