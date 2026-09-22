#!/usr/bin/env bash
# Submit NewTask FT for the VSA. Edit ../newtask_ft_common_config.yaml (dataset
# and source checkpoints) and vsa_ft_config.yaml (schedule), then run this.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${NEWTASK_FT_VSA_CONFIG:-${SCRIPT_DIR}/vsa_ft_config.yaml}"
if [ "$#" -ne 0 ]; then
  echo "Usage: $0   (the architecture is inherited from warm_start.vsa_checkpoint)" >&2
  exit 2
fi

# Freeze the component, shared, and global YAML for this submission.
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"   # sbatch, or a local run where the server has no Slurm
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/newtask_ft_vsa_config.py" --config "${CONFIG_PATH}" --shell
)"; then
  echo "NewTask FT VSA configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit NewTask FT VSA (${ARCHITECTURE_LABEL})"
echo "  warm start: ${VSA_CHECKPOINT_PATH}"
echo "  dataset   : ${SKILLVLA_DATASET_DIR}"
echo "  trainable : Cond + bottleneck + bridge + last ${TRAINABLE_EXPERT_LAYERS} Expert layer(s); DINO frozen=${FREEZE_VISION_ENCODER}"
echo "  run       : ${RUN_NAME}"
echo "  output    : ${OUTPUT_DIR}"

if [ "${NEWTASK_FT_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: sbatch ${SBATCH_ARGS[*]} ${SRC_DIR}/train.sbatch"
  exit 0
fi

NEWTASK_FT_VSA_CONFIG="${CONFIG_PATH}" \
NEWTASK_FT_VSA_RESOLVER="${SRC_DIR}/newtask_ft_vsa_config.py" \
NEWTASK_FT_BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON}" \
  submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
