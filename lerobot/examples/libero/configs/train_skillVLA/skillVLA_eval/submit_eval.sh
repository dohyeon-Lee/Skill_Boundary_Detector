#!/usr/bin/env bash
# Submit SkillVLA closed-loop EVAL (PT) on LIBERO sim.
#   (login) resolve config + check the PT checkpoint → sbatch eval.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # skillVLA_eval
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${SKILLVLA_EVAL_CONFIG:-${SCRIPT_DIR}/skillVLA_eval_config.yaml}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/skillVLA_eval_config.py" --config "${CONFIG_PATH}" --shell)"

if [ ! -d "${POLICY_PATH}" ]; then
  echo "PT checkpoint not found: ${POLICY_PATH}" >&2
  echo "Train it first: configs/train_skillVLA/skillVLA/submit_train.sh" >&2
  exit 1
fi
if [ ! -f "${FSQ_CKPT}" ]; then
  echo "FSQ checkpoint not found: ${FSQ_CKPT}" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit SkillVLA EVAL (PT)"
echo "  policy   : ${POLICY_PATH}"
echo "  target   : ${TARGET_TASK}  task_ids=${TASK_IDS}"
echo "  out      : ${EVAL_OUT_DIR}"
echo "  slurm    : partition=${EVAL_PARTITION} qos=${EVAL_QOS} gres=${EVAL_GRES} mem=${EVAL_MEM}"

SKILLVLA_EVAL_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
