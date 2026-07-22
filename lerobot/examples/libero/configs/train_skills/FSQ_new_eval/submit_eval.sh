#!/usr/bin/env bash
# Submit FSQ_new A/B/C closed-loop LIBERO evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FSQ_NEW_EVAL_CONFIG:-${SCRIPT_DIR}/fsq_new_eval_config.yaml}"

lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${lib}/snapshot_config.sh" ]; do lib="$(dirname "${lib}")"; done
source "${lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
mkdir -p "${SCRIPT_DIR}/logs"
FSQ_NEW_EVAL_ENV_SNAPSHOT="${SCRIPT_DIR}/logs/fsq_new_eval_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_new_eval_config.py" --config "${CONFIG_PATH}" --shell > "${FSQ_NEW_EVAL_ENV_SNAPSHOT}"
source "${FSQ_NEW_EVAL_ENV_SNAPSHOT}"
export FSQ_NEW_EVAL_ENV_SNAPSHOT

# Episode reset mapping is source-only. Keep an FSQ-local copy so jobs never depend on a SkillVLA run.
if [ ! -f "${EVAL_INIT_STATES_PATH}" ]; then
  mkdir -p "$(dirname "${EVAL_INIT_STATES_PATH}")"
  if [ -f "${LEGACY_EVAL_INIT_STATES_PATH}" ]; then
    cp "${LEGACY_EVAL_INIT_STATES_PATH}" "${EVAL_INIT_STATES_PATH}"
    echo "Copied source init-state map -> ${EVAL_INIT_STATES_PATH}"
  else
    echo "Building source init-state map from raw LeRobot + original LIBERO demos"
    "${BOOTSTRAP_PYTHON}" \
      "${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skillVLA/stage1_eval/oracle_matching/build_init_states.py" \
      --lerobot_dataset "${RAW_DATASET_DIR}" \
      --orig_dataset "${ORIGINAL_DATASET_DIR}" \
      --out "${EVAL_INIT_STATES_PATH}"
  fi
fi

# Encode each unique (checkpoint, skillset) once. A/B/C panels share this cache.
PREPARE_LINES="$("${BOOTSTRAP_PYTHON}" -c '
import json, os
seen=set()
for panel in json.loads(os.environ["MODELS_JSON"]):
    item=(panel["fsq_path"],panel["skills_dir"],panel["latents_path"])
    if item not in seen:
        seen.add(item); print("|".join(item))
')"
while IFS='|' read -r FSQ_PATH SKILLS_DIR LATENTS_PATH; do
  [ -n "${FSQ_PATH}" ] || continue
  PYTHONPATH="${SRC_DIR}:${PROJECT_ROOT}/lerobot/src" \
    "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/prepare_eval_data.py" \
      --model_path "${FSQ_PATH}" --skills_dir "${SKILLS_DIR}" \
      --output_path "${LATENTS_PATH}" --device cpu
done <<< "${PREPARE_LINES}"

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

make_chunks() {
  "${BOOTSTRAP_PYTHON}" - "${TASK_IDS}" "$1" <<'PY'
import json, sys
ids, n = json.loads(sys.argv[1]), max(1, int(sys.argv[2]))
n = min(n, len(ids))
base, rem, start = len(ids) // n, len(ids) % n, 0
for index in range(n):
    end = start + base + (1 if index < rem else 0)
    chunk = ids[start:end]; start = end
    tag = "" if n == 1 else f"t{chunk[0]}-{chunk[-1]}"
    print(f"{tag}|{json.dumps(chunk, separators=(',', ':'))}")
PY
}

TASKS_TOTAL="$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "${TASK_IDS}")"
PANEL_KEYS="$("${BOOTSTRAP_PYTHON}" -c 'import json,os; print("\n".join(x["key"] for x in json.loads(os.environ["MODELS_JSON"])))')"
N_PANELS="$(printf '%s\n' "${PANEL_KEYS}" | wc -l)"
CHUNK_COUNT=$(( ${EVAL_NUM_GPUS:-1} / N_PANELS ))
[ "${CHUNK_COUNT}" -lt 1 ] && CHUNK_COUNT=1
CHUNKS="$(make_chunks "${CHUNK_COUNT}")"
EVAL_FANOUT=""
index=0
while IFS='|' read -r TAG CHUNK; do
  [ -n "${CHUNK}" ] || continue
  while IFS= read -r KEY; do
    [ -n "${KEY}" ] || continue
    EVAL_FANOUT+="${KEY}|${CHUNK}|${TAG}"$'\n'
    echo "  [_${index}] ${KEY}${TAG:+_${TAG}}: ${CHUNK}"
    index=$((index + 1))
  done <<< "${PANEL_KEYS}"
done <<< "${CHUNKS}"

echo "Submit FSQ_new eval"
echo "  panels : ${N_PANELS}"
echo "  tasks  : ${TASK_IDS}"
echo "  output : ${EVAL_OUT_DIR}"
echo "  GPUs   : <=${EVAL_NUM_GPUS}"
mkdir -p "${EVAL_OUT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "  mode   : sequential srun in allocation ${SLURM_JOB_ID}"
  while IFS= read -r KEY; do
    [ -n "${KEY}" ] || continue
    PANEL_KEY="${KEY}" TASK_TAG="" TASKS_TOTAL="${TASKS_TOTAL}" \
      FSQ_NEW_EVAL_DIR="${SCRIPT_DIR}" FSQ_NEW_EVAL_CONFIG="${CONFIG_PATH}" \
      srun "${SRC_DIR}/eval.sbatch"
  done <<< "${PANEL_KEYS}"
else
  ARRAY_SPEC="0-$((index - 1))%${EVAL_NUM_GPUS}"
  echo "  mode   : sbatch --array=${ARRAY_SPEC}"
  FSQ_NEW_EVAL_DIR="${SCRIPT_DIR}" FSQ_NEW_EVAL_CONFIG="${CONFIG_PATH}" \
    EVAL_FANOUT="${EVAL_FANOUT}" TASKS_TOTAL="${TASKS_TOTAL}" \
    sbatch --job-name="FSQNeval" --array="${ARRAY_SPEC}" \
      --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err \
      "${SBATCH_ARGS[@]}" "${SRC_DIR}/eval.sbatch"
fi
