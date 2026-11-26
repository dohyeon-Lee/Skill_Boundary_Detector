#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
SHARED="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/src"
export SBD_GLOBAL_CONFIG="${SCRIPT_DIR}/../../../global_config.yaml"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
source "${SHARED}/submit_args.sh"
CONFIG="${DROID_FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"
MODE=submit
while (( $# )); do
  case "$1" in
    --config) CONFIG="${2:?--config requires a YAML path}"; shift 2 ;;
    --dry-run) MODE=plan; shift ;;
    --local) MODE=local; shift ;;
    *) echo "Usage: $0 [--JOB_ID|--yonsei] [--config FILE] [--dry-run|--local]" >&2; exit 2 ;;
  esac
done
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ "${MODE}" == plan ]]; then
  exec "${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}"
fi
# Validate before copying configs or requesting any GPU work.
"${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --shell >/dev/null
source "${SHARED}/snapshot_config.sh"
CONFIG="$(snapshot_config "${CONFIG}")"
export SBD_GLOBAL_CONFIG="$(dirname "${CONFIG}")/global_config.yaml"
"${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --freeze
SETTINGS="$("${PYTHON}" "${SCRIPT_DIR}/src/config.py" --config "${CONFIG}" --shell)"
eval "${SETTINGS}"
unset DROID_FSQ_MODELS_JSON  # workers resolve their own pinned snapshot
export DROID_FSQ_EVAL_DIR="${SCRIPT_DIR}" DROID_FSQ_EVAL_CONFIG="${CONFIG}"
echo "Report: ${EVAL_OUTPUT_DIR}"
if [[ "${MODE}" == local ]]; then
  exec bash "${SCRIPT_DIR}/src/run_eval.sbatch"
fi
source "${SHARED}/submit_job.sh"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
OPTIONS=(--job-name=DROID_FSQ_eval --cpus-per-task="${DROID_FSQ_EVAL_CPUS}"
  --mem="${DROID_FSQ_EVAL_MEMORY}" --time="${DROID_FSQ_EVAL_TIME}")
[[ "${DROID_FSQ_EVAL_DEVICE}" != cuda ]] || OPTIONS+=(--gres=gpu:1)
[[ -z "${DROID_FSQ_EVAL_PARTITION}" ]] || OPTIONS+=(--partition="${DROID_FSQ_EVAL_PARTITION}")
[[ -z "${DROID_FSQ_EVAL_QOS}" ]] || OPTIONS+=(--qos="${DROID_FSQ_EVAL_QOS}")
[[ -z "${DROID_FSQ_EVAL_NODELIST}" ]] || OPTIONS+=(--nodelist="${DROID_FSQ_EVAL_NODELIST}")
[[ -z "${DROID_FSQ_EVAL_EXCLUDE}" ]] || OPTIONS+=(--exclude="${DROID_FSQ_EVAL_EXCLUDE}")
submit_job "${OPTIONS[@]}" "${SCRIPT_DIR}/src/run_eval.sbatch"
