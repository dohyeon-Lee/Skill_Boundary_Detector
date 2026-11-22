#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Submit ABC SkillVLA Stage-1 VSA training using the shared LIBERO trainer.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
ABC_CONFIGS="${PROJECT_ROOT}/lerobot/skillVLA/abc/configs"
SHARED_PT="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skillVLA/PT"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SCRIPT_DIR}/vsa_train_config.yaml}"

source "${ABC_CONFIGS}/src/snapshot_config.sh"
source "${ABC_CONFIGS}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SHARED_PT}/src/stage1_train_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${BOOTSTRAP_EXPORTS}"

: "${SKILLVLA_DATASET_DIR:?ABC Stage-1 resolver did not export its dataset}"
SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
  --job-name=ABC_SkillVLA_PT
)
if [ -n "${TRAIN_NODELIST}" ]; then SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}"); fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}"); fi

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit ABC SkillVLA PT"
echo "  dataset : ${SKILLVLA_DATASET_DIR}"
echo "  jitter  : pmax=${TRANSITION_JITTER_PMAX} ${TRANSITION_JITTER_DISTRIBUTION}"
echo "  cameras : ${SKILL_IMAGE_KEYS}"
echo "  output  : ${PT_OUTPUT_DIR}"

# Existing allocations read this shell directly. Keep a private copy so later
# edits cannot change the script while Bash waits for the training process.
STAGE1_JOB_SCRIPT="$(dirname "${CONFIG_PATH}")/train.sbatch"
cp "${SHARED_PT}/src/train.sbatch" "${STAGE1_JOB_SCRIPT}"
bash -n "${STAGE1_JOB_SCRIPT}"

STAGE1_TRAIN_CONFIG="${CONFIG_PATH}" \
STAGE1_SRC_DIR="${SHARED_PT}/src" \
  submit_job "${SBATCH_ARGS[@]}" "${STAGE1_JOB_SCRIPT}"

