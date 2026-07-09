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

# Freeze the resolved env to a per-submit snapshot the JOB sources verbatim (no job-side emitter re-run
# on a possibly deleted/edited yaml). Exported → carried to all srun/sbatch(-array) paths via --export=ALL.
# Failure surfaces HERE at submit (set -e aborts), not as a confusing job-side traceback.
mkdir -p "${SCRIPT_DIR}/logs"
STAGE2_EVAL_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/stage2_eval_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage2_eval_config.py" --config "${CONFIG_PATH}" --shell > "${STAGE2_EVAL_ENV_SNAPSHOT}"
source "${STAGE2_EVAL_ENV_SNAPSHOT}"
export STAGE2_EVAL_ENV_SNAPSHOT

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

# task_ids → contiguous chunks ("TAG|CHUNK" lines; $1<=1 → one all-tasks line with tag "").
_make_chunks() {
  "${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "${1:-1}" <<'PY'
import json, sys
ids, n = json.loads(sys.argv[1]), max(1, int(sys.argv[2]))
if n <= 1:
    print(f"|{json.dumps(ids, separators=(',', ':'))}")
else:
    n = min(n, len(ids))                  # never more jobs than tasks
    base, rem = divmod(len(ids), n)
    s = 0
    for i in range(n):
        e = s + base + (1 if i < rem else 0)
        chunk = ids[s:e]; s = e
        print(f"t{chunk[0]}-{chunk[-1]}|{json.dumps(chunk, separators=(',', ':'))}")
PY
}

_preclean_chunks() {
  # Clean STALE chunk artifacts from a previous split of this same out dir, so this round's merge
  # (eval_info_t*.json count vs TASKS_TOTAL) and the merged-wandb sentinel see only THIS round.
  rm -f "${EVAL_OUT_DIR}"/eval_info_t*.json "${EVAL_OUT_DIR}"/task_success_rates_t*.png \
        "${EVAL_OUT_DIR}/.merged_wandb_done" \
        "${EVAL_OUT_DIR}"/panels/*/eval_info_t*.json "${EVAL_OUT_DIR}"/panels/*/task_success_rates_t*.png \
        "${EVAL_OUT_DIR}"/panels/*/.merged_wandb_done 2>/dev/null || true
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  # (eval_num_gpus / model fan-out are ignored here — one allocation = one sequential run.)
  [ "${EVAL_NUM_GPUS:-1}" -gt 1 ] && echo "  note     : eval_num_gpus=${EVAL_NUM_GPUS} ignored under srun (single allocation)"
  echo "  mode     : srun (reusing allocation ${SLURM_JOB_ID})"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ -n "${MODELS_JSON:-}" ]; then
  # ── MULTI-model: ONE job ARRAY ({ID}_0,{ID}_1,… — `scancel {ID}` kills the whole eval) with
  # tasks = MODELS × task-chunks (1 GPU each). The fan-out list is CHUNK-major (t-block0 × all
  # models first, then t-block1 × all models, …) so with limited GPUs every model advances the SAME
  # task block together — stage1-like: a task's grid clip appears as soon as its last panel job
  # finishes and stitches it (stitch/merge are idempotent+atomic across jobs).
  # eval_num_gpus = TOTAL GPU budget → chunks per model = eval_num_gpus // n_models (min 1). ──
  TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c "import json,sys; print(len(json.loads(sys.argv[1])))" "${TASK_IDS}")"
  LABELS="$("${BOOTSTRAP_PYTHON}" -c "import json,os; print('\n'.join(json.loads(os.environ['MODELS_LABELS'])))")"
  N_MODELS="$(printf '%s\n' "${LABELS}" | wc -l)"
  CHUNK_N=$(( ${EVAL_NUM_GPUS:-1} / N_MODELS ))
  [ "${CHUNK_N}" -lt 1 ] && CHUNK_N=1
  _preclean_chunks
  CHUNKS="$(_make_chunks "${CHUNK_N}")"
  EVAL_FANOUT=""
  i=0
  while IFS='|' read -r TAG CHUNK; do            # chunk-major interleave
    [ -z "${CHUNK}" ] && continue
    while IFS= read -r LBL; do
      [ -z "${LBL}" ] && continue
      EVAL_FANOUT+="${LBL}|${CHUNK}|${TAG}"$'\n'
      echo "    [_$i] ${LBL}${TAG:+_${TAG}}: task_ids=${CHUNK}"
      i=$((i + 1))
    done <<< "${LABELS}"
  done <<< "${CHUNKS}"
  echo "  mode     : sbatch --array 0-$((i - 1)) (models=${N_MODELS} × chunks=${CHUNK_N}, chunk-major, ≤${EVAL_NUM_GPUS} GPUs)"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    EVAL_FANOUT="${EVAL_FANOUT}" TASKS_TOTAL="${TASKS_TOTAL}" \
    sbatch --job-name="S2eval" --array="0-$((i - 1))" \
           --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
           "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS:-1}" -le 1 ]; then
  echo "  mode     : sbatch (new job)"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  # SINGLE model, eval_num_gpus > 1 → ONE job ARRAY over task chunks (each task = 1-GPU job).
  # All tasks share EVAL_OUT_DIR (task subdirs are disjoint); per-chunk eval_info_{tag}.json, a
  # wandb-run suffix, and the atomic FSQ_ft export keep them from clobbering each other.
  _preclean_chunks
  TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c "import json,sys; print(len(json.loads(sys.argv[1])))" "${TASK_IDS}")"
  CHUNKS="$(_make_chunks "${EVAL_NUM_GPUS:-1}")"
  EVAL_FANOUT=""
  i=0
  while IFS='|' read -r TAG CHUNK; do
    [ -z "${CHUNK}" ] && continue
    EVAL_FANOUT+="|${CHUNK}|${TAG}"$'\n'         # empty PANEL_LABEL = single-model
    echo "    [_$i] ${TAG}: task_ids=${CHUNK}"
    i=$((i + 1))
  done <<< "${CHUNKS}"
  echo "  mode     : sbatch --array 0-$((i - 1)) (task-split, 1 GPU each)"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    EVAL_FANOUT="${EVAL_FANOUT}" TASKS_TOTAL="${TASKS_TOTAL}" \
    sbatch --job-name="S2eval" --array="0-$((i - 1))" \
           --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
           "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
