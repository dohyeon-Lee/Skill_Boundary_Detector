#!/usr/bin/env bash
# Submit SkillVLA closed-loop EVAL (PT) on LIBERO sim.
#   (login) resolve config + check the PT checkpoint → sbatch eval.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # stage2_eval
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${STAGE2_EVAL_CONFIG:-${SCRIPT_DIR}/stage2_eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" --config "${CONFIG_PATH}" --shell)"

if [ ! -d "${POLICY_PATH}" ]; then
  echo "PT checkpoint not found: ${POLICY_PATH}" >&2
  echo "Train it first: configs/train_skillVLA/stage2/submit_train.sh" >&2
  exit 1
fi
if [ ! -f "${BASE_FSQ}" ]; then
  echo "Base FSQ not found: ${BASE_FSQ}  (the dataset's FSQ.pt the model was trained with)" >&2
  exit 1
fi

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

echo "Submit SkillVLA EVAL (PT)"
echo "  policy   : ${POLICY_PATH}"
echo "  target   : ${TARGET_TASK}  task_ids=${TASK_IDS}"
echo "  out      : ${EVAL_OUT_DIR}"
echo "  slurm    : partition=${EVAL_PARTITION} qos=${EVAL_QOS} gres=${EVAL_GRES} mem=${EVAL_MEM}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  # (eval_num_gpus is ignored here — one allocation = one sequential run.)
  [ "${EVAL_NUM_GPUS:-1}" -gt 1 ] && echo "  note     : eval_num_gpus=${EVAL_NUM_GPUS} ignored under srun (single allocation)"
  echo "  mode     : srun (reusing allocation ${SLURM_JOB_ID})"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS:-1}" -le 1 ]; then
  echo "  mode     : sbatch (new job)"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  # eval_num_gpus > 1 → split task_ids into N contiguous chunks, ONE 1-GPU job per chunk.
  # All jobs share EVAL_OUT_DIR (task subdirs are disjoint); per-chunk eval_info_{tag}.json, a
  # wandb-run suffix, and the atomic FSQ_ft export keep them from clobbering each other.
  echo "  mode     : sbatch ×${EVAL_NUM_GPUS} (task-split, 1 GPU each)"
  # Clean STALE chunk artifacts from a previous split of this same out dir, so this round's merge
  # (eval_info_t*.json count vs TASKS_TOTAL) and the merged-wandb sentinel see only THIS round.
  # (top level = single-model; panels/* = multi-model panel dirs)
  rm -f "${EVAL_OUT_DIR}"/eval_info_t*.json "${EVAL_OUT_DIR}"/task_success_rates_t*.png \
        "${EVAL_OUT_DIR}/.merged_wandb_done" \
        "${EVAL_OUT_DIR}"/panels/*/eval_info_t*.json "${EVAL_OUT_DIR}"/panels/*/task_success_rates_t*.png \
        "${EVAL_OUT_DIR}"/panels/*/.merged_wandb_done 2>/dev/null || true
  TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c "import json,sys; print(len(json.loads(sys.argv[1])))" "${TASK_IDS}")"
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
  while IFS='|' read -r TAG CHUNK; do
    [ -z "${CHUNK}" ] && continue
    echo "    job ${TAG}: task_ids=${CHUNK}"
    STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
      TASK_IDS="${CHUNK}" TASK_TAG="${TAG}" TASKS_TOTAL="${TASKS_TOTAL}" \
      sbatch --job-name="S2eval_${TAG}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
  done <<< "${CHUNKS}"
fi
