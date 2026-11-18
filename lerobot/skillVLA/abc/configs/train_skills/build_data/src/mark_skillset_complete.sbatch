#!/usr/bin/env bash
#SBATCH --job-name=ABC_mark_skillset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_BUILD_DATA_CONFIG:-${SUBMIT_DIR}/build_data_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SUBMIT_DIR}/../src/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"

[ -f "${SKILLSET_MANIFEST_PATH}" ] || { echo "Missing manifest: ${SKILLSET_MANIFEST_PATH}" >&2; exit 1; }
[ -f "${SKILLSET_GLOBAL_THRESHOLD_PATH}" ] || { echo "Missing global threshold." >&2; exit 1; }
DONE_COUNT="$(find "${SKILLSET_DIR}/skills" -name '*.done' -type f | wc -l)"
[ "${DONE_COUNT}" -eq "${ABC_EXPECTED_EPISODES}" ] || {
  echo "Incomplete ABC skillset: ${DONE_COUNT}/${ABC_EXPECTED_EPISODES} episode markers" >&2
  exit 1
}
SKILL_COUNT="$(find "${SKILLSET_DIR}/skills" -name '*.npz' -type f | wc -l)"
[ "${SKILL_COUNT}" -gt 0 ] || { echo "No skill files were generated." >&2; exit 1; }
{
  echo "dataset=${TARGET_DATASET}"
  echo "source_tasks=${ABC_SOURCE_TASKS_CSV}"
  echo "prompt_task_indices=${ABC_TASK_INDICES}"
  echo "episodes=${DONE_COUNT}"
  echo "skills=${SKILL_COUNT}"
  echo "skillset_mode=${SKILLSET_MODE}"
  echo "action_space=${SKILLSET_ACTION_MODE}_14d_bimanual"
  echo "relative_exclude_joints=${SKILLSET_RELATIVE_EXCLUDE_JOINTS}"
  echo "global_threshold_path=${SKILLSET_GLOBAL_THRESHOLD_PATH}"
  echo "completed_at=$(date -Iseconds)"
} > "${SKILLSET_DONE_PATH}"
echo "ABC skillset complete: ${SKILLSET_DONE_PATH} (${DONE_COUNT} episodes, ${SKILL_COUNT} skills)"
