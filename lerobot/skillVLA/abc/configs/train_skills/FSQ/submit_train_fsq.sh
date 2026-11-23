#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
# Submit ABC reconstructor-only FSQ training using fsq_config.yaml.
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../src/submit_args.sh"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ABC_FSQ_CONFIG:-${SCRIPT_DIR}/fsq_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
if ! CONFIG_EXPORTS="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../src/train_skills_config.py" \
  --config "${CONFIG_PATH}" --stage fsq --shell)"; then
  echo "ABC FSQ configuration failed; no job was submitted." >&2
  exit 1
fi
eval "${CONFIG_EXPORTS}"

[ -d "${SKILLSET_DIR}/skills" ] || {
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  exit 1
}

SBATCH_ARGS=(
  --partition="${FSQ_TRAIN_PARTITION}"
  --qos="${FSQ_TRAIN_QOS}"
  --gres="${FSQ_TRAIN_GRES}"
  --cpus-per-task="${FSQ_TRAIN_CPUS_PER_TASK}"
  --mem="${FSQ_TRAIN_MEM}"
  --time="${FSQ_TRAIN_TIME}"
)
[ -n "${FSQ_TRAIN_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${FSQ_TRAIN_NODELIST}")
[ -n "${FSQ_TRAIN_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${FSQ_TRAIN_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs_train
echo "Submit ABC FSQ"
echo "  tasks   : ${ABC_SOURCE_TASKS_LABEL}"
echo "  boundary: ${SKILLSET_SEG_NAME}"
echo "  action  : ${FSQ_ACTION_SPACE} ${FSQ_ACTION_DIM}D; grippers=[${FSQ_ACTION_GRIPPER_INDICES_STR}]"
echo "  output  : ${FSQ_OUTPUT_DIR}"
echo "  slurm   : partition=${FSQ_TRAIN_PARTITION} qos=${FSQ_TRAIN_QOS} gres=${FSQ_TRAIN_GRES}"

# Existing allocations run Bash directly from disk. Freeze the script bytes
# alongside the YAML so edits during a long training cannot corrupt Bash's
# read position when control returns from Python.
FSQ_EXEC_SCRIPT="$(dirname "${CONFIG_PATH}")/train_fsq.sbatch"
cp -- "${SCRIPT_DIR}/src/train_fsq.sbatch" "${FSQ_EXEC_SCRIPT}"
bash -n "${FSQ_EXEC_SCRIPT}"

ABC_FSQ_CONFIG="${CONFIG_PATH}" submit_job "${SBATCH_ARGS[@]}" "${FSQ_EXEC_SCRIPT}"
