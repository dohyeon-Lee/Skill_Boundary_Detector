#!/usr/bin/env bash
# Submit the Stage-2 VLM attention heatmap job (resolves slurm args from attn_viz_config.yaml +
# configs/global_config.yaml, then sbatches src/S2_attn_viz.sbatch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${ATTN_VIZ_CONFIG:-${SCRIPT_DIR}/attn_viz_config.yaml}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/attn_viz_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}" --qos="${EVAL_QOS}" --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}" --mem="${EVAL_MEM}" --time="${EVAL_TIME}"
)
[ -n "${EVAL_NODELIST}" ]      && SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -n "${EVAL_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

JID=$(ATTN_VIZ_CONFIG="${CONFIG_PATH}" sbatch --parsable "${SBATCH_ARGS[@]}" "${SRC_DIR}/S2_attn_viz.sbatch")
echo "attn_viz job ${JID} → ${OUTPUT_DIR}/index.html"
