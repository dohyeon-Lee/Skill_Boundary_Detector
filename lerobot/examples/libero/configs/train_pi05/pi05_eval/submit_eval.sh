#!/usr/bin/env bash
# Submit pi05 closed-loop eval on LIBERO sim.
#   SINGLE model → one job (or, with eval_num_gpus>1, a task-split ARRAY).
#   MULTI models (`models:` in the yaml) → a models×task-chunks ARRAY (chunk-major) → per-task videos
#   stitched into ONE labelled grid, task-by-task. Slurm settings from configs/global_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # pi05_eval
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # train_pi05
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_PI05_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_config.py"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

# Freeze the resolved env to a per-submit snapshot the JOB sources verbatim (no job-side emitter re-run
# on a possibly deleted/edited yaml). The per-array chunk override (TASK_IDS/TASK_TAG/PANEL_LABEL) is
# applied by eval.sbatch AFTER sourcing the snapshot, so it actually takes effect (unlike a pre-source
# override, which the snapshot would clobber).
mkdir -p "${SCRIPT_DIR}/logs"
PI05_EVAL_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/pi05_eval_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell > "${PI05_EVAL_ENV_SNAPSHOT}"
source "${PI05_EVAL_ENV_SNAPSHOT}"
export PI05_EVAL_ENV_SNAPSHOT

if [ -z "${MODELS_JSON:-}" ] && [ ! -d "${EVAL_POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${EVAL_POLICY_PATH}" >&2
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
[ -n "${EVAL_NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -n "${EVAL_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"

echo "Submit pi05 eval"
echo "  target  : ${EVAL_TARGET_TASK}  task_ids=${EVAL_TASK_IDS}"
echo "  out     : ${EVAL_OUT_DIR}"
echo "  slurm   : partition=${EVAL_PARTITION} nodelist=${EVAL_NODELIST:-<none>} exclude=${EVAL_EXCLUDE_NODES:-<none>}"

# task_ids → contiguous chunks ("TAG|CHUNK" lines; $1<=1 → one all-tasks line with tag "").
_make_chunks() {
  "${BOOTSTRAP_PYTHON}" - "${EVAL_TASK_IDS}" "${1:-1}" <<'PY'
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
  rm -f "${EVAL_OUT_DIR}"/eval_info_t*.json "${EVAL_OUT_DIR}/.merged_wandb_done" \
        "${EVAL_OUT_DIR}"/panels/*/eval_info_t*.json "${EVAL_OUT_DIR}"/panels/*/.merged_wandb_done 2>/dev/null || true
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (salloc) → reuse the held GPU as a job step (one sequential run;
  # eval_num_gpus / model fan-out ignored). Resources come from the allocation.
  [ "${EVAL_NUM_GPUS:-1}" -gt 1 ] && echo "  note    : eval_num_gpus=${EVAL_NUM_GPUS} ignored under srun (single allocation)"
  echo "  mode    : srun (reusing allocation ${SLURM_JOB_ID})"
  TRAIN_PI05_CONFIG="${CONFIG_PATH}" srun "${SCRIPT_DIR}/eval.sbatch"
elif [ -n "${MODELS_JSON:-}" ]; then
  # ── MULTI-model: ONE job ARRAY (tasks = MODELS × task-chunks, 1 GPU each). CHUNK-major fan-out
  # (t-block0 × all models first, then t-block1 × all models, …) so with limited GPUs every model
  # advances the SAME task block together — a task's grid clip appears as soon as its last panel job
  # finishes and stitches it. chunks per model = eval_num_gpus // n_models (min 1). ──
  TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c "import json,sys; print(len(json.loads(sys.argv[1])))" "${EVAL_TASK_IDS}")"
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
  echo "  mode    : sbatch --array 0-$((i - 1)) (models=${N_MODELS} × chunks=${CHUNK_N}, chunk-major, ≤${EVAL_NUM_GPUS} GPUs)"
  TRAIN_PI05_CONFIG="${CONFIG_PATH}" EVAL_FANOUT="${EVAL_FANOUT}" TASKS_TOTAL="${TASKS_TOTAL}" \
    sbatch --job-name="pi05_eval" --array="0-$((i - 1))" \
           --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
           "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS:-1}" -le 1 ]; then
  echo "  mode    : sbatch (new job)"
  TRAIN_PI05_CONFIG="${CONFIG_PATH}" sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/eval.sbatch"
else
  # SINGLE model, eval_num_gpus > 1 → ONE job ARRAY over task chunks (each = 1-GPU job). All chunks
  # share EVAL_OUT_DIR (disjoint task subdirs); per-chunk eval_info_{tag}.json + a merged finisher.
  _preclean_chunks
  TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c "import json,sys; print(len(json.loads(sys.argv[1])))" "${EVAL_TASK_IDS}")"
  CHUNKS="$(_make_chunks "${EVAL_NUM_GPUS:-1}")"
  EVAL_FANOUT=""
  i=0
  while IFS='|' read -r TAG CHUNK; do
    [ -z "${CHUNK}" ] && continue
    EVAL_FANOUT+="|${CHUNK}|${TAG}"$'\n'         # empty PANEL_LABEL = single-model
    echo "    [_$i] ${TAG}: task_ids=${CHUNK}"
    i=$((i + 1))
  done <<< "${CHUNKS}"
  echo "  mode    : sbatch --array 0-$((i - 1)) (task-split, 1 GPU each)"
  TRAIN_PI05_CONFIG="${CONFIG_PATH}" EVAL_FANOUT="${EVAL_FANOUT}" TASKS_TOTAL="${TASKS_TOTAL}" \
    sbatch --job-name="pi05_eval" --array="0-$((i - 1))" \
           --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
           "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/eval.sbatch"
fi
