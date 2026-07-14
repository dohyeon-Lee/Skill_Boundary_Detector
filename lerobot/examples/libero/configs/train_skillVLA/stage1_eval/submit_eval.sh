#!/usr/bin/env bash
# Submit SkillVLA Stage-1 (skill_expert) closed-loop oracle eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage1_eval
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE1_EVAL_CONFIG:-${SCRIPT_DIR}/stage1_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_eval_config.py" --config "${CONFIG_PATH}" --shell)"

REQUIRED_ARTIFACTS=("${FSQ_CKPT}" "${SKILL_LABEL_DATASET_DIR}")
[ "${PRIMARY_MODEL_KIND}" = "stage1" ] && REQUIRED_ARTIFACTS=("${POLICY_PATH}" "${REQUIRED_ARTIFACTS[@]}")
for p in "${REQUIRED_ARTIFACTS[@]}"; do
  if [ ! -e "${p}" ]; then
    echo "Missing artifact: ${p}" >&2
    exit 1
  fi
done

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
  --time="${EVAL_TIME}"
)
if [ -n "${EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
fi
if [ -n "${EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit Stage-1 oracle eval"
echo "  backend: ${PRIMARY_MODEL_KIND}"
echo "  policy : ${POLICY_PATH:-<FSQ expert-only>}"
echo "  fsq    : ${FSQ_CKPT}"
echo "  out    : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  # (eval_num_gpus is ignored here — one allocation = one sequential run.)
  [ "${EVAL_NUM_GPUS:-1}" -gt 1 ] && echo "  note   : eval_num_gpus=${EVAL_NUM_GPUS} ignored under srun (single allocation)"
  echo "  mode   : srun (reusing allocation ${SLURM_JOB_ID})"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS:-1}" -le 1 ]; then
  echo "  mode   : sbatch (new job)"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  # eval_num_gpus > 1 → ONE Slurm job array (JOBID_0, JOBID_1, ...), one task chunk/GPU.
  # All elements share EVAL_OUT_DIR but write disjoint task folders and tagged summaries.
  CHUNKS="$("${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "${EVAL_NUM_GPUS}" <<'PY'
import json, sys
ids, n = json.loads(sys.argv[1]), max(1, int(sys.argv[2]))
n = min(n, len(ids))                      # never more jobs than tasks
base, rem = divmod(len(ids), n)
s = 0
for i in range(n):
    e = s + base + (1 if i < rem else 0)
    chunk = ids[s:e]; s = e
    print(f"t{chunk[0]}-{chunk[-1]}|{json.dumps(chunk, separators=(',', ':'))}")
PY
)"
  EVAL_FANOUT=""
  i=0
  while IFS='|' read -r TAG CHUNK; do
    [ -z "${CHUNK}" ] && continue
    EVAL_FANOUT+="${CHUNK}|${TAG}"$'\n'
    echo "    [_${i}] ${TAG}: task_ids=${CHUNK}"
    i=$((i + 1))
  done <<< "${CHUNKS}"
  ARRAY_SPEC="0-$((i - 1))%${EVAL_NUM_GPUS}"
  echo "  mode   : sbatch --array ${ARRAY_SPEC} (task-split, 1 GPU each, ≤${EVAL_NUM_GPUS} GPUs)"
  STAGE1_EVAL_DIR="${SCRIPT_DIR}" STAGE1_EVAL_CONFIG="${CONFIG_PATH}" \
    EVAL_FANOUT="${EVAL_FANOUT}" \
    sbatch --job-name="S1eval" --array="${ARRAY_SPEC}" \
           --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
           "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
