#!/usr/bin/env bash
# Inputs:
#   skillset     : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/skills
# Reference model:
#   none
# Outputs:
#   completion   : {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/FSQ_inputs/seg_{dp}_ck{ckpt}/skillset/.complete
#
# Mark skillset generation complete after the array succeeds.
#SBATCH --job-name=mark_skillset
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

# Slurm executes a spool copy of this file. Resolve repository paths from the
# build_data directory where the submitter invoked sbatch, not from BASH_SOURCE.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${SUBMIT_DIR}/src"
COMMON_SRC_DIR="${SCRIPT_DIR}/../../src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}"

if [ ! -f "${SCRIPT_DIR}/mark_skillset_complete.sbatch" ]; then
  echo "Cannot resolve build_data/src from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${SUBMIT_DIR}"
mkdir -p logs

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" \
    --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --stage build_data --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" \
    --config "${CONFIG_PATH}" --stage build_data --shell)"
fi
eval "${RESOLVED_SETTINGS}"

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset skills dir missing: ${SKILLSET_DIR}/skills" >&2
  exit 1
fi
if [ ! -f "${SKILLSET_DIR}/skillset_manifest.json" ]; then
  echo "Skillset manifest missing: ${SKILLSET_DIR}/skillset_manifest.json" >&2
  exit 1
fi

EFFECTIVE_GLOBAL_THRESHOLD_PATH="${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-${SKILLSET_GLOBAL_THRESHOLD_PATH}}"
if [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ] && [ ! -f "${EFFECTIVE_GLOBAL_THRESHOLD_PATH}" ]; then
  echo "Global threshold file missing: ${EFFECTIVE_GLOBAL_THRESHOLD_PATH}" >&2
  exit 1
fi

# A one-GPU supervisor may finish only after every expected episode/task pair
# has its completion marker, including filtered and too-short episodes.
if [ "${LIBERO_SINGLE_GPU_BUILD:-false}" = true ]; then
  "${BOOTSTRAP_PYTHON}" - "${PROJECT_ROOT}" "${RAW_DATASET_DIR}" "${SKILLSET_DIR}" "${LIBERO_EXPECTED_EPISODES}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "lerobot/skillVLA/libero/configs/train_skillVLA/build_data/src"))
from verify_skillset import task_to_episodes
mapping = task_to_episodes(Path(sys.argv[2]), None)
root = Path(sys.argv[3]) / "skills"
expected = {(ep, task) for task, episodes in mapping.items() for ep in episodes}
if len(expected) != int(sys.argv[4]):
    raise RuntimeError("Dataset episode/task mapping changed since submission.")
missing = [(ep, task) for ep, task in expected
           if not (root / f"task{task:02d}" / f"ep{ep:05d}_task{task:02d}.done").is_file()]
if missing:
    raise RuntimeError(f"Incomplete LIBERO skillset: {len(missing)}/{len(expected)} missing episodes; {missing[:10]}")
print(f"Verified all {len(expected)} episode completion records.")
PY
fi

N_SKILLS=$(find "${SKILLSET_DIR}/skills" -name '*.npz' | wc -l)
if [ "${N_SKILLS}" -eq 0 ]; then
  echo "No skill npz files found under ${SKILLSET_DIR}/skills" >&2
  exit 1
fi

mkdir -p "${SKILLSET_DIR}"
{
  echo "dataset=${TARGET_DATASET}"
  echo "skills=${N_SKILLS}"
  [ -z "${LIBERO_EXPECTED_EPISODES:-}" ] || echo "episodes=${LIBERO_EXPECTED_EPISODES}"
  echo "skillset_mode=${SKILLSET_MODE}"
  echo "manifest=${SKILLSET_DIR}/skillset_manifest.json"
  echo "boundary_threshold_mode=${SKILLSET_BOUNDARY_THRESHOLD_MODE}"
  echo "boundary_threshold_scale=${SKILLSET_BOUNDARY_THRESHOLD_SCALE}"
  [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "global_mean" ] && \
    echo "global_threshold_path=${EFFECTIVE_GLOBAL_THRESHOLD_PATH}"
  [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ] && \
    echo "global_threshold_source=${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
  echo "completed_at=$(date -Iseconds)"
} > "${SKILLSET_DONE_PATH}"

echo "Skillset complete: ${SKILLSET_DONE_PATH} (${N_SKILLS} skills)"
