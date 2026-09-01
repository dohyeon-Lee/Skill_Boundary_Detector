#!/usr/bin/env bash
# Resolve one frozen YAML and build its per-episode exact-init map.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ORACLE_MATCHING_CONFIG:-${SCRIPT_DIR}/oracle_matching_config.yaml}"
if [ "${1:-}" = "--config" ]; then
  [ $# -ge 2 ] || { echo "--config requires a YAML path" >&2; exit 2; }
  CONFIG_PATH="$2"
  shift 2
elif [ $# -gt 0 ]; then
  if [[ "$1" == *.yaml ]] || [ -f "$1" ]; then
    CONFIG_PATH="$1"
  else
    # Backward-compatible source override for the former run.sh interface.
    export ORACLE_SOURCE_DATASET="$1"
  fi
  shift
fi
[ $# -eq 0 ] || { echo "Unexpected arguments: $*" >&2; exit 2; }

PROJECT_HINT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
BOOTSTRAP_PYTHON="${PROJECT_HINT}/.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || {
  echo "Project venv Python not found: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
}

ORACLE_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/oracle_matching_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${ORACLE_EXPORTS}"

if [ -f "${ORACLE_OUTPUT_PATH}" ] && [ "${ORACLE_OVERWRITE}" != true ]; then
  echo "Oracle init-state map already exists; no-op: ${ORACLE_OUTPUT_PATH}"
  exit 0
fi

mkdir -p "$(dirname "${ORACLE_OUTPUT_PATH}")"

if [ "${ORACLE_MODE}" = langgap ]; then
  if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_RENDER:-0}" != 1 ]; then
    echo "LangGap matching renders simulator candidates and must run in a GPU allocation." >&2
    echo "Use ./submit_oracle_matching.sh (or set ALLOW_LOGIN_RENDER=1 intentionally)." >&2
    exit 1
  fi
  command=(
    "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/build_langgap_init_states.py"
    --lerobot_dataset "${ORACLE_LEROBOT_DATASET}"
    --out "${ORACLE_OUTPUT_PATH}"
    --signature-size "${ORACLE_SIGNATURE_SIZE}"
    --num-steps-wait "${ORACLE_NUM_STEPS_WAIT}"
    --state-weight "${ORACLE_STATE_WEIGHT}"
    --image-weight "${ORACLE_IMAGE_WEIGHT}"
    --wrist-weight "${ORACLE_WRIST_WEIGHT}"
    --max-state-score "${ORACLE_MAX_STATE_SCORE}"
    --max-image-mae "${ORACLE_MAX_IMAGE_MAE}"
    --min-score-margin "${ORACLE_MIN_SCORE_MARGIN}"
  )
  [ -z "${ORACLE_TASK_IDS}" ] || command+=(--task-indices "${ORACLE_TASK_IDS}")
  [ -z "${ORACLE_MAX_EPISODES}" ] || command+=(--max-episodes "${ORACLE_MAX_EPISODES}")
  [ -z "${ORACLE_CACHE_DIR}" ] || command+=(--cache-dir "${ORACLE_CACHE_DIR}")
  [ "${ORACLE_ACCEPT_AMBIGUOUS}" != true ] || command+=(--accept-ambiguous)
  [ "${ORACLE_OVERWRITE}" != true ] || command+=(--overwrite)
else
  command=(
    "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/build_init_states.py"
    --lerobot_dataset "${ORACLE_LEROBOT_DATASET}"
    --orig_dataset "${ORACLE_ORIGINAL_DATASET}"
    --out "${ORACLE_OUTPUT_PATH}"
  )
fi

echo "Build oracle init-state map"
echo "  mode    : ${ORACLE_MODE}"
echo "  dataset : ${ORACLE_LEROBOT_DATASET}"
echo "  output  : ${ORACLE_OUTPUT_PATH}"
exec "${command[@]}"
