#!/usr/bin/env bash
# Submit SkillVLA FINETUNING (FT) (policy.type=skill_vla, warm-started from a Stage-2 checkpoint).
#   (login) resolve config + check artifacts → sbatch train.sbatch
# Prereqs: the new task's skillvla dataset (configs/train_skillVLA/build_data) and a trained Stage-2
# checkpoint (configs/train_skillVLA/stage2, named by stage2_run_name in the yaml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # FT
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FT_TRAIN_CONFIG:-${SCRIPT_DIR}/ft_train_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Resolve the config → FREEZE the resolved env to a per-submit snapshot the JOB sources verbatim, so the
# job NEVER re-runs the emitter on a (possibly deleted/edited) yaml at start. Emitter failure surfaces
# HERE at submit (set -e aborts), not as a confusing job-side traceback.
mkdir -p "${SCRIPT_DIR}/logs"
FT_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/ft_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_train_config.py" --config "${CONFIG_PATH}" --shell > "${FT_ENV_SNAPSHOT}"
source "${FT_ENV_SNAPSHOT}"

# ── prerequisite artifacts ──
if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
  exit 1
fi
if [ ! -e "${STAGE2_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-2 checkpoint: ${STAGE2_CHECKPOINT_PATH}" >&2
  echo "Train Stage-2 first: configs/train_skillVLA/stage2/submit_train.sh (set stage2_run_name in the yaml)" >&2
  exit 1
fi
if [ ! -e "${STAGE1_CHECKPOINT_PATH}" ]; then
  echo "Missing Stage-1 checkpoint (architecture config): ${STAGE1_CHECKPOINT_PATH}" >&2
  exit 1
fi
# (train_terminator DINO 토큰 존재 검사 은퇴 — terminator는 배치 현재 프레임을 ONLINE 토큰화.)

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

echo "Submit SkillVLA FT"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  warmstart: ${STAGE2_CHECKPOINT_PATH}"
echo "  cond/term: ${COND_SKILL_SOURCE} / ${TRAIN_TERMINATOR}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job step.
  echo "  mode     : srun (reusing allocation ${SLURM_JOB_ID})"
  FT_TRAIN_CONFIG="${CONFIG_PATH}" FT_ENV_SNAPSHOT="${FT_ENV_SNAPSHOT}" \
    srun "${SRC_DIR}/train.sbatch"
else
  echo "  mode     : sbatch (new job)"
  FT_TRAIN_CONFIG="${CONFIG_PATH}" FT_ENV_SNAPSHOT="${FT_ENV_SNAPSHOT}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
fi
