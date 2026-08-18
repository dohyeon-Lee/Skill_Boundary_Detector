#!/usr/bin/env bash
# Submit clean Stage-2 evaluation; eval_num_gpus caps how many jobs (GPUs) run,
# and (task x panel) units are packed into that many chunks.
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
)

# slurm.time budgets one (task x panel) unit; a chunked job needs that budget
# once per unit it carries.
scale_time() {
  "${BOOTSTRAP_PYTHON}" - "$1" "$2" <<'PY'
import math, sys

spec, factor = sys.argv[1].strip(), int(sys.argv[2])
if factor <= 1:
    print(spec)
    raise SystemExit
days = 0
if "-" in spec:
    day_part, rest = spec.split("-", 1)
    days = int(day_part)
    parts = [int(p) for p in rest.split(":")]
    parts += [0] * (3 - len(parts))  # days-H[:M[:S]]
    hours, minutes, seconds = parts
else:
    parts = [int(p) for p in spec.split(":")]
    if len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0  # bare minutes
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts
total_seconds = (((days * 24 + hours) * 60 + minutes) * 60 + seconds) * factor
total_minutes = math.ceil(total_seconds / 60)
out_days, rem = divmod(total_minutes, 24 * 60)
out_hours, out_minutes = divmod(rem, 60)
print(f"{out_days}-{out_hours:02d}:{out_minutes:02d}:00")
PY
}
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit Stage-2 eval"
echo "  panels    : ${MODEL_COUNT}"
echo "  policy    : ${POLICY_PATH}"
echo "  predictor : ${EXTERNAL_PREDICTOR_MODEL:-<none>}"
echo "  terminator: ${EXTERNAL_TERMINATOR_MODEL:-<none>}"
echo "  output    : ${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : srun in allocation ${SLURM_JOB_ID}"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS}" -le 1 ]; then
  UNITS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c \
    'import json, sys; print(len(json.loads(sys.argv[1])) * max(1, int(sys.argv[2])))' \
    "${TASK_IDS}" "${MODEL_COUNT}")"
  JOB_TIME="$(scale_time "${EVAL_TIME}" "${UNITS_TOTAL}")"
  echo "  mode   : one sbatch job (${UNITS_TOTAL} task x panel units)"
  echo "  time   : ${EVAL_TIME} x ${UNITS_TOTAL} units -> ${JOB_TIME}"
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --time="${JOB_TIME}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
else
  # Pack (task x panel) units into at most eval_num_gpus chunks: one array
  # element per chunk, so each job loads its models once and walks its share
  # sequentially. With eval_num_gpus >= units this degenerates to the old
  # one-unit-per-element fanout (maximum parallelism). First output line is
  # the largest chunk's unit count, used to scale the per-job time limit.
  FANOUT_RAW="$("${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "${MODEL_COUNT}" "${EVAL_NUM_GPUS}" <<'PY'
import json, sys

task_ids = json.loads(sys.argv[1])
panels = max(1, int(sys.argv[2]))
n_gpus = max(1, int(sys.argv[3]))


def split(seq, k):
    """Split seq into k (capped at len) near-equal contiguous groups."""
    k = max(1, min(k, len(seq)))
    base, rem = divmod(len(seq), k)
    groups, start = [], 0
    for index in range(k):
        size = base + (1 if index < rem else 0)
        groups.append(seq[start : start + size])
        start += size
    return groups


chunks = []
if n_gpus >= panels:
    # Hand each panel its share of the job slots, then split its tasks over
    # them; every chunk stays a single-panel task group.
    slot_base, slot_rem = divmod(min(n_gpus, len(task_ids) * panels), panels)
    for panel in range(panels):
        slots = slot_base + (1 if panel < slot_rem else 0)
        for group in split(task_ids, slots):
            chunks.append((group, [panel]))
else:
    # Fewer jobs than panels: each job takes every task for a panel group.
    for panel_group in split(list(range(panels)), n_gpus):
        chunks.append((list(task_ids), panel_group))


def tag(ids, ps):
    t = f"t{ids[0]}" if len(ids) == 1 else f"t{ids[0]}-{ids[-1]}"
    p = f"p{ps[0]:02d}" if len(ps) == 1 else f"p{ps[0]:02d}-{ps[-1]:02d}"
    return f"{t}_{p}"


print(max(len(ids) * len(ps) for ids, ps in chunks))
for ids, ps in chunks:
    ids_json = json.dumps(ids, separators=(",", ":"))
    print(f"{ids_json}|{tag(ids, ps)}|{','.join(str(p) for p in ps)}")
PY
)"
  MAX_UNITS="$(printf '%s\n' "${FANOUT_RAW}" | head -n 1)"
  CHUNKS="$(printf '%s\n' "${FANOUT_RAW}" | tail -n +2)"
  EVAL_FANOUT="${CHUNKS}"$'\n'
  ARRAY_SIZE="$(printf '%s\n' "${CHUNKS}" | sed '/^$/d' | wc -l)"
  ARRAY_SPEC="0-$((ARRAY_SIZE - 1))%${EVAL_NUM_GPUS}"
  CHUNK_TIME="$(scale_time "${EVAL_TIME}" "${MAX_UNITS}")"
  echo "  mode   : array ${ARRAY_SPEC} (${ARRAY_SIZE} chunks, <=${MAX_UNITS} task x panel units each)"
  echo "  time   : ${EVAL_TIME} x ${MAX_UNITS} units -> ${CHUNK_TIME}"
  export EVAL_FANOUT
  STAGE2_EVAL_DIR="${SCRIPT_DIR}" STAGE2_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --job-name=S2eval --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      --time="${CHUNK_TIME}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
