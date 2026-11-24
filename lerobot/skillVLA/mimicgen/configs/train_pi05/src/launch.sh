#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
# Sourced by the train/eval launchers with HERE, CONFIG, SELECTOR and PREFIX.
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
RESOURCE_ROOT="${CODE_ROOT}"
if [ ! -x "${RESOURCE_ROOT}/.venv/bin/python" ]; then
  COMMON_GIT="$(git -C "${CODE_ROOT}" rev-parse --git-common-dir)"
  [[ "${COMMON_GIT}" = /* ]] || COMMON_GIT="${CODE_ROOT}/${COMMON_GIT}"
  RESOURCE_ROOT="$(cd "${COMMON_GIT}/.." && pwd)"
fi
PYTHON="${RESOURCE_ROOT}/.venv/bin/python"
export PATH="${RESOURCE_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${CODE_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
SHARED="${CODE_ROOT}/lerobot/skillVLA/libero/configs"
source "${SHARED}/src/submit_args.sh"
DRY_RUN=false
while (( $# )); do
  case "$1" in
    --config) CONFIG="${2:?missing YAML}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "${DRY_RUN}" = true ]; then
  exec "${PYTHON}" "${HERE}/src/config.py" --config "${CONFIG}"
fi
SNAPSHOT="${HERE}/.config_snapshots/$(date +%Y%m%d-%H%M%S)_$$_${RANDOM}"
EXPORTS="$("${PYTHON}" "${HERE}/src/config.py" --config "${CONFIG}" --freeze "${SNAPSHOT}" --shell)"
eval "${EXPORTS}"
export "${SELECTOR}=${FROZEN_CONFIG}"
export SBD_SERVER="${SERVER}"
export SBD_GLOBAL_CONFIG="${SNAPSHOT}/global_config.yaml"
source "${SHARED}/src/submit_job.sh"
PARTITION_VAR="${PREFIX}_PARTITION"; QOS_VAR="${PREFIX}_QOS"; GRES_VAR="${PREFIX}_GRES"
CPUS_VAR="${PREFIX}_CPUS_PER_TASK"; MEM_VAR="${PREFIX}_MEM"; TIME_VAR="${PREFIX}_TIME"
NODE_VAR="${PREFIX}_NODELIST"; EXCLUDE_VAR="${PREFIX}_EXCLUDE_NODES"
ARGS=(--partition="${!PARTITION_VAR}" --qos="${!QOS_VAR}" --gres="${!GRES_VAR}"
      --cpus-per-task="${!CPUS_VAR}" --mem="${!MEM_VAR}" --time="${!TIME_VAR}")
[ -z "${!NODE_VAR}" ] || ARGS+=(--nodelist="${!NODE_VAR}")
[ -z "${!EXCLUDE_VAR}" ] || ARGS+=(--exclude="${!EXCLUDE_VAR}")
cd "${HERE}"
mkdir -p logs
echo "Frozen configuration: ${FROZEN_CONFIG}"
submit_job "${ARGS[@]}" "${HERE}/src/${BATCH_SCRIPT}"
