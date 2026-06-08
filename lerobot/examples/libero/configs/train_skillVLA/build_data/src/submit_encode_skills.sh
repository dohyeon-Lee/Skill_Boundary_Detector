#!/usr/bin/env bash
# Step 4 of SkillVLA data generation: map each segmented skill to an FSQ latent vector.
#   (sbatch) extract skill-level DINO tokens, then encode them with the trained FSQ.
#
# Inputs (from train_skillVLA_config.yaml + submit_build_skillset.sh outputs):
#   skillset         : {skillvla_dataset}/{source}/_work/seg_{dp}_ck{ckpt}/skillset/skills/
#   per-episode DINO : {skillvla_dataset}/{source}/_work/dino/pg{grid}/ (manifest + episode npz)
#   FSQ model        : {project_root}/FSQ_outputs/{fsq_run_name}/FSQ_epoch{ckpt:04d}.pt (or FSQ.pt)
# Outputs (intermediate, merged into skillvla later):
#   skill DINO tokens: {skillvla_dataset}/{source}/_work/seg_{dp}_ck{ckpt}/skill_tokens.npz
#   skill latents    : {skillvla_dataset}/{source}/{run_tag}/skill_latents.npz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/../train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

mkdir -p "${SCRIPT_DIR}/../logs"
SNAPSHOT_ENV="${SCRIPT_DIR}/../logs/skillvla_env_encode_$(date +%Y%m%d_%H%M%S)_$$.sh"
if [ -n "${SOURCE_DATASET}" ]; then
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell > "${SNAPSHOT_ENV}"
else
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell > "${SNAPSHOT_ENV}"
fi
source "${SNAPSHOT_ENV}"

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run submit_build_skillset.sh first." >&2
  exit 1
fi
if [ ! -d "${DINO_PER_EPISODE_DIR}" ]; then
  echo "Per-episode DINO not found: ${DINO_PER_EPISODE_DIR}" >&2
  echo "Run submit_build_skillset.sh first (it slices the DINO)." >&2
  exit 1
fi
if [ ! -f "${FSQ_MODEL_PATH}" ]; then
  echo "FSQ model not found: ${FSQ_MODEL_PATH}" >&2
  echo "Train the FSQ first (configs/train_skills/FSQ) or fix fsq_* in the config." >&2
  exit 1
fi

# ── skip if the encode outputs already exist ──
if [ -f "${SKILL_TOKENS_PATH}" ] && [ -f "${SKILL_LATENTS_PATH}" ]; then
  echo "Skill tokens + latents already exist → nothing to do."
  echo "  ${SKILL_TOKENS_PATH}"
  echo "  ${SKILL_LATENTS_PATH}"
  exit 0
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

echo "Submit FSQ skill encoding"
echo "  source    : ${SOURCE_DATASET}"
echo "  skillset  : ${SKILLSET_DIR}/skills"
echo "  dino dir  : ${DINO_PER_EPISODE_DIR}"
echo "  FSQ model : ${FSQ_MODEL_PATH}"
echo "  tokens    : ${SKILL_TOKENS_PATH}"
echo "  latents   : ${SKILL_LATENTS_PATH}"

TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" SKILLVLA_ENV_SNAPSHOT="${SNAPSHOT_ENV}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/encode_skills.sbatch"
