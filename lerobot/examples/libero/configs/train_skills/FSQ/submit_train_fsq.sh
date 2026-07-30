#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   skillset     : selected by the five folder components in fsq_config.yaml;
#                  DP/detector provenance is read from skillset_manifest.json
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/outputs/FSQ/{fsq_run_name}
#   FSQ checkpoints: {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt
#   skill tokens : {project_root}/outputs/FSQ/{fsq_run_name}/skill_latents.npz
#
# Submit FSQ training using Slurm values from train_skills_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/fsq_config.yaml}"

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
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "${RESOLVED_SETTINGS}"

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run build_data/submit_build_data.sh first." >&2
  exit 1
fi
# FSQ terminator images are decoded live at the sampled timesteps (no feature warm-pass or cache —
# dino-prep 빌드 단계 제거). raw dataset의 videos/만 있으면 됨.
if [ ! -d "${RAW_DATASET_DIR}/videos" ]; then
  echo "Raw dataset videos not found: ${RAW_DATASET_DIR}/videos" >&2
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
echo "  DINO (online): raw ← ${RAW_DATASET_DIR}/videos (warm-pass at train start)"
echo "  output      : ${FSQ_OUTPUT_DIR}"
echo "  slurm       : partition=${FSQ_TRAIN_PARTITION} qos=${FSQ_TRAIN_QOS} gres=${FSQ_TRAIN_GRES}"

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${FSQ_SRC_DIR}/train_fsq.sbatch"
