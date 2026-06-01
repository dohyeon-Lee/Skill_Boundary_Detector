#!/usr/bin/env bash
# Step 5 (final) of SkillVLA data generation. One sbatch does:
#   (a) merge per-episode DINO → dino.npz             (SkillVLA visual input, train-time npz)
#   (b) add_skill_latents_to_dataset → skillvla/      (LeRobot dataset + skill columns)
#   (c) copy the FSQ checkpoint → FSQ.pt
#   (d) optionally clean up intermediates (_work, skill_latents.npz)
#
# Final outputs in {skillvla_dataset}/{source}/{run_tag}/ :
#   dino.npz   FSQ.pt   skillvla/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/../train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

# ── skip if the final outputs already exist (checked first: intermediates may have
#    been cleaned up after a prior successful run) ──
if [ -f "${DINO_NPZ_PATH}" ] && [ -f "${FSQ_COPY_PATH}" ] \
   && [ -d "${SKILLVLA_DATASET_DIR}" ] && [ -n "$(ls -A "${SKILLVLA_DATASET_DIR}" 2>/dev/null)" ]; then
  echo "SkillVLA outputs already exist → nothing to do (${SKILLVLA_RUN_DIR})"
  exit 0
fi

if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Source dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -f "${SKILL_LATENTS_PATH}" ]; then
  echo "skill_latents.npz not found: ${SKILL_LATENTS_PATH}" >&2
  echo "Run submit_encode_skills.sh first." >&2
  exit 1
fi
if [ ! -d "${DINO_PER_EPISODE_DIR}" ]; then
  echo "Per-episode DINO not found: ${DINO_PER_EPISODE_DIR}" >&2
  echo "Run submit_build_skillset.sh first." >&2
  exit 1
fi
if [ ! -f "${FSQ_MODEL_PATH}" ]; then
  echo "FSQ model not found: ${FSQ_MODEL_PATH}" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${SKILLVLA_PARTITION}"
  --qos="${SKILLVLA_QOS}"
  --gres="${SKILLVLA_GRES}"
  --cpus-per-task="${SKILLVLA_CPUS_PER_TASK}"
  --mem="${SKILLVLA_MEM}"
  --time="${SKILLVLA_TIME}"
)
if [ -n "${SKILLVLA_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${SKILLVLA_NODELIST}")
fi
if [ -n "${SKILLVLA_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${SKILLVLA_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}/.."
mkdir -p logs

echo "Submit SkillVLA dataset build"
echo "  source    : ${SOURCE_DATASET}"
echo "  latents   : ${SKILL_LATENTS_PATH}"
echo "  dino.npz  : ${DINO_NPZ_PATH}"
echo "  FSQ.pt    : ${FSQ_COPY_PATH}"
echo "  skillvla  : ${SKILLVLA_DATASET_DIR}"
echo "  cleanup   : ${CLEANUP_INTERMEDIATE}"

TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillvla.sbatch"
