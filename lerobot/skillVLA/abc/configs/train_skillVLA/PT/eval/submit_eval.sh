#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${CONFIGS_DIR}/../../../.." && pwd)"
BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${ABC_SKILLVLA_EVAL_CONFIG:-${SCRIPT_DIR}/pt_eval_config.yaml}"
SHARED_SRC="${CONFIGS_DIR}/train_pi05/PT/eval/src"

source "${CONFIGS_DIR}/src/snapshot_config.sh"
source "${CONFIGS_DIR}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
PIN_ARGS=()
if [ "${ABC_EVAL_DRY_RUN:-0}" != 1 ]; then
  PIN_ARGS+=(--pin-models)
fi
CONFIG_PATH="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/freeze_config.py" --config "${CONFIG_PATH}" "${PIN_ARGS[@]}")"
eval "$("${BOOTSTRAP_PYTHON}" "${SHARED_SRC}/eval_config.py" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(--job-name=ABC_SkillVLA_eval --partition="${ABC_EVAL_PARTITION}" --qos="${ABC_EVAL_QOS}"
  --gres="${ABC_EVAL_GRES}" --cpus-per-task="${ABC_EVAL_CPUS}" --mem="${ABC_EVAL_MEMORY}" --time="${ABC_EVAL_TIME}")
[ -z "${ABC_EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${ABC_EVAL_NODELIST}")
[ -z "${ABC_EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${ABC_EVAL_EXCLUDE_NODES}")
export ABC_EVAL_DIR="${SCRIPT_DIR}" ABC_EVAL_SRC_DIR="${SCRIPT_DIR}/src"
export ABC_EVAL_CONFIG="${CONFIG_PATH}" ABC_EVAL_PROJECT_ROOT="${PROJECT_ROOT}"
cd "${SCRIPT_DIR}"
mkdir -p logs
echo "ABC SkillVLA evaluation: VSA + configured GT/Predictor conditions + Term_Token_1"
echo "  tasks  : ${ABC_EVAL_TASKS_JSON}"
echo "  output : ${ABC_EVAL_OUTPUT_DIR}"
echo "  config : ${CONFIG_PATH} (checkpoint steps pinned)"
if [ "${ABC_EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: nothing submitted."
  exit 0
fi
submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/eval.sbatch"
