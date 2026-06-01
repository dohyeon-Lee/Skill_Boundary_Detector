#!/usr/bin/env bash
# Submit pi05 FT using Slurm settings from ../train_pi05_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_PI05_CONFIG:-${ROOT_DIR}/train_pi05_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

SBATCH_ARGS=(
  --partition="${FT_PARTITION}"
  --qos="${FT_QOS}"
  --gres="${FT_GRES}"
  --cpus-per-task="${FT_CPUS_PER_TASK}"
  --mem="${FT_MEM}"
  --time="${FT_TIME}"
)
[ -n "${FT_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${FT_NODELIST}")
[ -n "${FT_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${FT_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit pi05 FT"
echo "  dataset : ${FT_DATASET_DIR}"
echo "  base    : ${FT_PRETRAINED_MODEL_PATH}"
echo "  output  : ${FT_OUTPUT_DIR}"
echo "  slurm   : partition=${FT_PARTITION} nodelist=${FT_NODELIST:-<none>} exclude=${FT_EXCLUDE_NODES:-<none>}"

TRAIN_PI05_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/pi05_FT.sbatch"
