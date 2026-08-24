#!/usr/bin/env bash
# Submit FT evaluation; (task x panel) units are packed across eval_num_gpus.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FT_EVAL_CONFIG:-${SCRIPT_DIR}/ft_eval_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

# FT config resolution is stdlib-only. Avoid touching the shared project venv
# until the compute job stages it onto node-local scratch.
BOOTSTRAP_PYTHON=/usr/bin/python3
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_eval_config.py" --config "${CONFIG_PATH}" --shell)"

for artifact in "${POLICY_PATH}" "${FSQ_PATH}" "${SKILL_DATASET_DIR}"; do
  [ -e "${artifact}" ] || { echo "Missing FT eval artifact: ${artifact}" >&2; exit 1; }
done

# The FT sbatch delegates to the shared Stage-2 runner, so export the archive
# under the runner's environment name. Disable with FT_EVAL_NODE_LOCAL_VENV=0.
FT_EVAL_VENV_ARCHIVE="${FT_EVAL_VENV_ARCHIVE:-}"
if [ "${FT_EVAL_NODE_LOCAL_VENV:-1}" = "1" ]; then
  source "${SCRIPT_DIR}/../../node_local_venv.sh"
  if [ -z "${FT_EVAL_VENV_ARCHIVE}" ] && \
     ! FT_EVAL_VENV_ARCHIVE="$(prepare_node_local_venv_archive "${PROJECT_ROOT}" "FT eval venv")"; then
    FT_EVAL_VENV_ARCHIVE=""
    echo "FT eval venv: preparation failed; jobs will use the shared venv." >&2
  fi
else
  FT_EVAL_VENV_ARCHIVE=""
fi
export STAGE2_EVAL_VENV_ARCHIVE="${FT_EVAL_VENV_ARCHIVE}"
export STAGE2_EVAL_VENV_LABEL="FT eval venv"

SBATCH_ARGS=(
  --partition="${EVAL_PARTITION}"
  --qos="${EVAL_QOS}"
  --gres="${EVAL_GRES}"
  --cpus-per-task="${EVAL_CPUS_PER_TASK}"
  --mem="${EVAL_MEM}"
)
[ -z "${EVAL_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${EVAL_NODELIST}")
[ -z "${EVAL_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${EVAL_EXCLUDE_NODES}")

scale_time() {
  "${BOOTSTRAP_PYTHON}" - "$1" "$2" <<'PY'
import math
import sys

spec, factor = sys.argv[1].strip(), int(sys.argv[2])
if factor <= 1:
    print(spec)
    raise SystemExit
days = 0
if "-" in spec:
    day_part, rest = spec.split("-", 1)
    days = int(day_part)
    parts = [int(part) for part in rest.split(":")]
    parts += [0] * (3 - len(parts))
    hours, minutes, seconds = parts
else:
    parts = [int(part) for part in spec.split(":")]
    if len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts
total = (((days * 24 + hours) * 60 + minutes) * 60 + seconds) * factor
total_minutes = math.ceil(total / 60)
out_days, remainder = divmod(total_minutes, 24 * 60)
out_hours, out_minutes = divmod(remainder, 60)
print(f"{out_days}-{out_hours:02d}:{out_minutes:02d}:00")
PY
}

cd "${SCRIPT_DIR}"
mkdir -p logs
echo "Submit Stage-2 FT eval"
echo "  panels    : ${MODEL_COUNT}"
echo "  policy    : ${POLICY_PATH}"
echo "  predictor : ${EXTERNAL_PREDICTOR_MODEL:-<none>}"
echo "  terminator: ${EXTERNAL_TERMINATOR_MODEL:-<none>}"
echo "  output    : ${EVAL_OUT_DIR}"
if [ -n "${FT_EVAL_VENV_ARCHIVE}" ]; then
  echo "  Python    : node-local copy from ${FT_EVAL_VENV_ARCHIVE}"
else
  echo "  Python    : shared ${PROJECT_ROOT}/.venv"
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode      : srun in allocation ${SLURM_JOB_ID}"
  FT_EVAL_DIR="${SCRIPT_DIR}" FT_EVAL_CONFIG="${CONFIG_PATH}" \
    srun "${SRC_DIR}/eval.sbatch"
elif [ "${EVAL_NUM_GPUS}" -le 1 ]; then
  UNITS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c \
    'import json, sys; print(len(json.loads(sys.argv[1])) * max(1, int(sys.argv[2])))' \
    "${TASK_IDS}" "${MODEL_COUNT}")"
  JOB_TIME="$(scale_time "${EVAL_TIME}" "${UNITS_TOTAL}")"
  echo "  mode      : one sbatch job (${UNITS_TOTAL} task x panel units)"
  echo "  time      : ${EVAL_TIME} x ${UNITS_TOTAL} units -> ${JOB_TIME}"
  FT_EVAL_DIR="${SCRIPT_DIR}" FT_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --job-name=FTeval --time="${JOB_TIME}" "${SBATCH_ARGS[@]}" \
      "${SRC_DIR}/eval.sbatch"
else
  FANOUT_RAW="$("${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "${MODEL_COUNT}" "${EVAL_NUM_GPUS}" <<'PY'
import json
import sys

task_ids = json.loads(sys.argv[1])
panels = max(1, int(sys.argv[2]))
n_gpus = max(1, int(sys.argv[3]))


def split(sequence, count):
    count = max(1, min(count, len(sequence)))
    base, remainder = divmod(len(sequence), count)
    groups, start = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        groups.append(sequence[start : start + size])
        start += size
    return groups


chunks = []
if n_gpus >= panels:
    slot_base, slot_remainder = divmod(
        min(n_gpus, len(task_ids) * panels), panels
    )
    for panel in range(panels):
        slots = slot_base + (1 if panel < slot_remainder else 0)
        for group in split(task_ids, slots):
            chunks.append((group, [panel]))
else:
    for panel_group in split(list(range(panels)), n_gpus):
        chunks.append((list(task_ids), panel_group))


def tag(ids, panel_ids):
    task = f"t{ids[0]}" if len(ids) == 1 else f"t{ids[0]}-{ids[-1]}"
    panel = (
        f"p{panel_ids[0]:02d}"
        if len(panel_ids) == 1
        else f"p{panel_ids[0]:02d}-{panel_ids[-1]:02d}"
    )
    return f"{task}_{panel}"


print(max(len(ids) * len(panel_ids) for ids, panel_ids in chunks))
for ids, panel_ids in chunks:
    ids_json = json.dumps(ids, separators=(",", ":"))
    selected = ",".join(str(panel) for panel in panel_ids)
    print(f"{ids_json}|{tag(ids, panel_ids)}|{selected}")
PY
)"
  MAX_UNITS="$(printf '%s\n' "${FANOUT_RAW}" | head -n 1)"
  CHUNKS="$(printf '%s\n' "${FANOUT_RAW}" | tail -n +2)"
  EVAL_FANOUT="${CHUNKS}"$'\n'
  ARRAY_SIZE="$(printf '%s\n' "${CHUNKS}" | sed '/^$/d' | wc -l)"
  ARRAY_SPEC="0-$((ARRAY_SIZE - 1))%${EVAL_NUM_GPUS}"
  CHUNK_TIME="$(scale_time "${EVAL_TIME}" "${MAX_UNITS}")"
  echo "  mode      : array ${ARRAY_SPEC} (${ARRAY_SIZE} chunks, <=${MAX_UNITS} units each)"
  echo "  time      : ${EVAL_TIME} x ${MAX_UNITS} units -> ${CHUNK_TIME}"
  export EVAL_FANOUT
  FT_EVAL_DIR="${SCRIPT_DIR}" FT_EVAL_CONFIG="${CONFIG_PATH}" \
    sbatch --job-name=FTeval --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      --time="${CHUNK_TIME}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
