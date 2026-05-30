#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   FSQ inputs   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/skillset/skills
#   DINO tokens  : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/dino_tokens_pg{dino_patch_grid}.npz
#   SAM2 flags   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/patch_flags.npz
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/FSQ_outputs/{fsq_run_name}
#   FSQ model    : {project_root}/FSQ_outputs/{fsq_run_name}/FSQ.pt
#   skill tokens : {project_root}/FSQ_outputs/{fsq_run_name}/skill_latents.npz
#
# Submit FSQ training using Slurm values from train_skills_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/../train_skills_config.yaml}"
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
  echo "Run FSQ/submit_prepare_fsq_inputs.sh first." >&2
  exit 1
fi
if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "DINO tokens not found: ${DINO_TOKENS_PATH}" >&2
  echo "Run FSQ/submit_prepare_fsq_inputs.sh first." >&2
  exit 1
fi
if [ "${DECODER_IMAGE_MODE}" = "dino_flags" ] && [ ! -f "${SAM2_FLAGS_PATH}" ] && [ ! -d "${SAM2_MASKS_DIR}" ]; then
  echo "Patch flags not found for decoder_image_mode=dino_flags:" >&2
  echo "  ${SAM2_FLAGS_PATH}" >&2
  echo "Run FSQ/submit_prepare_fsq_inputs.sh with fsq_build_patch_flags: true." >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${FSQ_TRAIN_PARTITION}"
  --qos="${FSQ_TRAIN_QOS}"
  --gres="${FSQ_TRAIN_GRES}"
  --cpus-per-task="${FSQ_TRAIN_CPUS_PER_TASK}"
  --mem="${FSQ_TRAIN_MEM}"
  --time="${FSQ_TRAIN_TIME}"
)

if [ -n "${FSQ_TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_TRAIN_NODELIST}")
fi
if [ -n "${FSQ_TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs_train

echo "Submit FSQ train"
echo "  dataset     : ${TARGET_DATASET}"
echo "  FSQ inputs  : ${FSQ_INPUTS_DIR}"
echo "  skillset    : ${SKILLSET_DIR}/skills"
echo "  DINO tokens : ${DINO_TOKENS_PATH}"
echo "  SAM2 flags  : ${SAM2_FLAGS_PATH}"
echo "  output      : ${FSQ_OUTPUT_DIR}"
echo "  slurm       : partition=${FSQ_TRAIN_PARTITION} qos=${FSQ_TRAIN_QOS} gres=${FSQ_TRAIN_GRES}"

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${FSQ_SRC_DIR}/train_fsq.sbatch"
