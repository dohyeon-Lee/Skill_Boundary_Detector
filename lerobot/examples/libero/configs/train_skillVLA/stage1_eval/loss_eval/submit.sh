#!/usr/bin/env bash
# Submit the offline front-vs-back SKILL loss eval on one GPU (Slurm). Models + sampling come from
# loss_eval_config.yaml; Slurm settings (partition/qos/gres/...) from loss_eval_config.py (+ global_config).
# Inside an salloc → srun (reuse the held GPU); otherwise → sbatch a fresh job. Extra args → loss_eval.py:
#     ./submit.sh                     # yaml's n_batches × batch_size per model
#     ./submit.sh --n_batches 150     # override the sampling
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../stage1_eval/loss_eval
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"    # SBD (7 up)
CONFIG_PATH="${LOSS_EVAL_CONFIG:-${HERE}/loss_eval_config.yaml}"

BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.venv/bin/python"          # emitter deps: yaml only (torch-free)
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

eval "$("${BOOTSTRAP_PYTHON}" "${HERE}/loss_eval_config.py" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -n "${EVAL_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -n "${EVAL_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${HERE}"; mkdir -p logs
export LOSS_EVAL_ARGS="$*"
echo "Submit loss-eval (models from ${CONFIG_PATH})  args='${LOSS_EVAL_ARGS}'"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode: srun (reusing allocation ${SLURM_JOB_ID})"
  LOSS_EVAL_DIR="${HERE}" srun "${HERE}/loss_eval.sbatch"
else
  echo "  mode: sbatch (new job)"
  LOSS_EVAL_DIR="${HERE}" sbatch "${SBATCH_ARGS[@]}" "${HERE}/loss_eval.sbatch"
fi
