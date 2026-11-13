#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${ABC_SIM_EVAL_CONFIG:-${SCRIPT_DIR}/abc_sim_eval_config.yaml}"
CONFIGS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd "${CONFIGS_DIR}/../../../.." && pwd)"
BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

source "${CONFIGS_DIR}/src/snapshot_config.sh"
source "${CONFIGS_DIR}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/eval_config.py" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --job-name=abc_sim_eval
  --partition="${ABC_EVAL_PARTITION}"
  --qos="${ABC_EVAL_QOS}"
  --gres="${ABC_EVAL_GRES}"
  --cpus-per-task="${ABC_EVAL_CPUS}"
  --mem="${ABC_EVAL_MEMORY}"
  --time="${ABC_EVAL_TIME}"
)
[ -z "${ABC_EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${ABC_EVAL_NODELIST}")
[ -z "${ABC_EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${ABC_EVAL_EXCLUDE_NODES}")

export ABC_EVAL_DIR="${SCRIPT_DIR}"
export ABC_EVAL_SRC_DIR="${SRC_DIR}"
export ABC_EVAL_CONFIG="${CONFIG_PATH}"
export ABC_EVAL_PROJECT_ROOT="${PROJECT_ROOT}"

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit ABC Sim eval"
echo "  tasks  : ${ABC_EVAL_TASKS_JSON}"
echo "  output : ${ABC_EVAL_OUTPUT_DIR}"
echo "  slurm  : partition=${ABC_EVAL_PARTITION} qos=${ABC_EVAL_QOS} exclude=${ABC_EVAL_EXCLUDE_NODES:-<none>}"

if [ "${ABC_EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: nothing submitted."
  exit 0
fi

submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
