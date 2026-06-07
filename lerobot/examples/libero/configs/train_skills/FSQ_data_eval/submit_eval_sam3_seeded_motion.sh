#!/usr/bin/env bash
# Inputs:
#   dataset    : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   skillset   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
#   raw video  : {project_root}/{dataset_root}/{target_dataset}/videos
# Reference models:
#   SAM3       : {project_root}/models/sam3
#   SAM2       : sam2_checkpoint or {project_root}/models/sam2/sam2.1_hiera_large.pt
# Outputs:
#   PNGs       : FSQ_data_eval/outputs/sam3_seeded_motion by default
#
# Submit SAM3-seeded motion visualization as a GPU Slurm job.
# Example:
#   ./submit_eval_sam3_seeded_motion.sh --task_ids 0 --n_episodes 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../train_skills_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run FSQ/submit_prepare_fsq_inputs.sh first, or pass --skills_dir explicitly." >&2
  exit 1
fi

EVAL_ARGS_QUOTED=""
printf -v EVAL_ARGS_QUOTED "%q " "$@"

PARTITION="${FSQ_DATA_EVAL_PARTITION:-${SLURM_PARTITION}}"
QOS="${FSQ_DATA_EVAL_QOS:-${SLURM_QOS}}"
GRES="${FSQ_DATA_EVAL_GRES:-${SLURM_GRES}}"
CPUS="${FSQ_DATA_EVAL_CPUS_PER_TASK:-8}"
MEM="${FSQ_DATA_EVAL_MEM:-80G}"
TIME="${FSQ_DATA_EVAL_TIME:-06:00:00}"
NODELIST="${FSQ_DATA_EVAL_NODELIST:-${SLURM_NODELIST}}"
EXCLUDE="${FSQ_DATA_EVAL_EXCLUDE_NODES:-${SLURM_EXCLUDE_NODES}}"

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --qos="${QOS}"
  --gres="${GRES}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME}"
)

if [ -n "${NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi
if [ -n "${EXCLUDE}" ]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit SAM3-seeded motion eval"
echo "  dataset   : ${TARGET_DATASET:-${TARGET_DATASET}}"
echo "  skillset  : ${SKILLSET_DIR}/skills"
echo "  raw data  : ${RAW_DATASET_DIR}"
echo "  slurm     : partition=${PARTITION} qos=${QOS} gres=${GRES} mem=${MEM}"
echo "  args      : ${EVAL_ARGS_QUOTED}"

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" EVAL_ARGS_QUOTED="${EVAL_ARGS_QUOTED}" \
  sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/eval_sam3_seeded_motion.sbatch"
