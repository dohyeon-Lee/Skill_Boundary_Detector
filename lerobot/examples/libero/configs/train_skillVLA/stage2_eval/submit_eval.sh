#!/usr/bin/env bash
# Submit clean Stage-2 evaluation; eval_num_gpus splits task ids over a Slurm array.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE2_EVAL_CONFIG:-${SCRIPT_DIR}/stage2_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" --config "${CONFIG_PATH}" --shell)"

for artifact in "${POLICY_PATH}" "${FSQ_PATH}" "${SKILL_DATASET_DIR}"; do
  [ -e "${artifact}" ] || { echo "Missing artifact: ${artifact}" >&2; exit 1; }
done

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit Stage-2 eval"
echo "  models : ${MODEL_COUNT}"
echo "  policy : ${POLICY_PATH}"
echo "  output : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun in allocation ${SLURM_JOB_ID}"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS}" -le 1 ]; then
  echo "  mode   : one sbatch job"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  CHUNKS="$("${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "${EVAL_NUM_GPUS}" <<'PY'
import json, sys
task_ids = json.loads(sys.argv[1])
jobs = min(max(1, int(sys.argv[2])), len(task_ids))
base, extra = divmod(len(task_ids), jobs)
start = 0
for index in range(jobs):
    end = start + base + (index < extra)
    chunk = task_ids[start:end]
    start = end
    print(json.dumps(chunk, separators=(",", ":")) + f"|t{chunk[0]}-{chunk[-1]}")
PY
)"
  EVAL_FANOUT="${CHUNKS}"$'\n'
  ARRAY_SIZE="$(printf '%s\n' "${CHUNKS}" | sed '/^$/d' | wc -l)"
  ARRAY_SPEC="0-$((ARRAY_SIZE - 1))%${EVAL_NUM_GPUS}"
  echo "  mode   : array ${ARRAY_SPEC}"
  printf '  chunks :\n%s\n' "${CHUNKS}"
  export EVAL_FANOUT
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --job-name=S2eval --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
