#!/usr/bin/env bash
# Stage 4: encode a target skillset with a trained FSQ model for SkillVLA.
#
# Inputs:
#   - FSQ_MODEL_PATH: trained FSQ checkpoint (.pt)
#   - DATA: target dataset name whose skillset/DINO tokens will be encoded
#
# Output:
#   libero_dataset/${DATA}_data/${DATA}_for_skillVLA/${OUTPUT_NAME}/
#     ├── skill_latents.npz
#     └── FSQ.pt

#SBATCH --job-name=FSQ_skillVLA_encode
# SBATCH --partition=debug
# SBATCH --nodelist=node100
# SBATCH --qos=big_qos

#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
  CONFIG_PY=${SUBMIT_DIR}/../pipeline_config.py
fi
if [ ! -f "${CONFIG_PY}" ]; then
  echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
  echo "Submit from lerobot root or from examples/libero/configs/data_generation/4_skillVLA." >&2
  exit 1
fi
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell)"

cd "${HOMEDIR}${PROJDIR}/lerobot"
source "${HOMEDIR}${PROJDIR}/.venv/bin/activate"
unset LD_LIBRARY_PATH

FSQ_MODEL_PATH=${FSQ_SOURCE_CKPT}
OUTPUT_DIR=${FSQ_PACKAGE_DIR}
LATENTS_PATH=${SKILL_LATENTS_PATH}
MODEL_COPY_PATH=${FSQ_CKPT}

if [ "${ZERO_START_EEF}" = "true" ]; then
  ZERO_START_ARG="--zero_start_eef"
else
  ZERO_START_ARG="--no-zero_start_eef"
fi

echo "FSQ SkillVLA encoding"
echo "  dataset : ${DATA}"
echo "  model   : ${FSQ_MODEL_PATH}"
echo "  skills  : ${SKILLSET_DIR}/skills"
echo "  DINO    : ${DINO_TOKENS_PATH}"
echo "  output  : ${OUTPUT_DIR}"
echo "  eef mode: $([ "${ZERO_START_EEF}" = "true" ] && echo zero || echo abs)"

if [ ! -f "${FSQ_MODEL_PATH}" ]; then
  echo "ERROR: FSQ model not found: ${FSQ_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "ERROR: skillset directory not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run data_generation/2_skillset and 3_FSQ for DATA=${DATA} first." >&2
  exit 1
fi

if [ ! -f "${DINO_TOKENS_PATH}" ]; then
  echo "ERROR: DINO token file not found: ${DINO_TOKENS_PATH}" >&2
  echo "Run data_generation/3_FSQ/run_dino_parallel.sh for DATA=${DATA} first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python examples/libero/encode_FSQ_skills.py \
    --model_path      ${FSQ_MODEL_PATH} \
    --skills_dir      ${SKILLSET_DIR}/skills \
    --dino_features   ${DINO_TOKENS_PATH} \
    --output_path     ${LATENTS_PATH} \
    ${ZERO_START_ARG} \
    --device          cuda

cp -f "${FSQ_MODEL_PATH}" "${MODEL_COPY_PATH}"

echo "DONE"
echo "  latents: ${LATENTS_PATH}"
echo "  model  : ${MODEL_COPY_PATH}"
