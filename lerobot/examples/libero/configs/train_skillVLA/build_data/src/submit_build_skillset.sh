#!/usr/bin/env bash
# Steps 2-3 of SkillVLA data generation, in one submission:
#   (login)  slice per-episode DINO for the source dataset from the base full DINO
#   (sbatch) run the trained DP to segment the source dataset into skills (skillset)
#
# Inputs (from train_skillVLA_config.yaml):
#   source dataset : {project_root}/{dataset_root}/{source_dataset}
#   base DINO      : {project_root}/{dataset_root}/{dino_base_dataset}_DINO/pg{grid}/
#   DP policy      : {project_root}/DP_outputs/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
# Outputs (intermediate, removed at end of full pipeline):
#   per-episode DINO : {skillvla_dataset}/{source}/_work/dino/pg{grid}/
#   skillset         : {skillvla_dataset}/{source}/_work/seg_{dp}_ck{ckpt}/skillset/skills/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/../train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Source dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -d "${DP_POLICY_PATH}" ]; then
  echo "DP policy checkpoint not found: ${DP_POLICY_PATH}" >&2
  echo "Train the DP first (configs/train_skills/DP) or fix dp_policy_name/dp_checkpoint." >&2
  exit 1
fi

# ── skip if the skillset is already complete ──
if "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/verify_skillset.py" \
     --dataset_dir "${RAW_DATASET_DIR}" --skillset_dir "${SKILLSET_DIR}" --check >/dev/null 2>&1; then
  echo "Skillset already complete → nothing to do (${SKILLSET_DIR})"
  exit 0
fi

# ── compute the task-shard array (one shard = SKILLSET_TASKS_PER_JOB tasks → 1 GPU) ──
ALL_TASK_IDS="$("${BOOTSTRAP_PYTHON}" - "${RAW_DATASET_DIR}/meta/tasks.parquet" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1]).reset_index()
print(" ".join(str(int(t)) for t in sorted(df["task_index"].unique())))
PY
)"
N_TASKS=$(wc -w <<< "${ALL_TASK_IDS}")
EXPECTED_EPISODES=$("${BOOTSTRAP_PYTHON}" - "${RAW_DATASET_DIR}/meta/episodes" <<'PY'
import sys
from pathlib import Path
import pandas as pd
parts = sorted(Path(sys.argv[1]).rglob("file-*.parquet"))
print(sum(len(pd.read_parquet(p)) for p in parts))
PY
)
TPJ="${SKILLSET_TASKS_PER_JOB:-5}"
NUM_SHARDS=$(( (N_TASKS + TPJ - 1) / TPJ ))
[ "${NUM_SHARDS}" -lt 1 ] && NUM_SHARDS=1
ARRAY_SPEC="0-$(( NUM_SHARDS - 1 ))"
if [ "${SKILLSET_ARRAY_THROTTLE:-0}" -gt 0 ]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${SKILLSET_ARRAY_THROTTLE}"
fi

# ── boundary mode: direct episode segmentation or global two-pass reduction ──
SBATCH_ARGS=(
  --partition="${SKILLVLA_PARTITION}"
  --qos="${SKILLVLA_QOS}"
  --gres="${SKILLVLA_GRES}"
  --cpus-per-task="${SKILLVLA_CPUS_PER_TASK}"
  --mem="${SKILLVLA_MEM}"
  --time="${SKILLVLA_TIME}"
)
if [ -n "${SKILLVLA_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${SKILLVLA_NODELIST}")
fi
if [ -n "${SKILLVLA_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${SKILLVLA_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}/.."
mkdir -p logs

echo "Submit DP skillset (${SKILLSET_BOUNDARY_THRESHOLD_MODE})"
echo "  source    : ${SOURCE_DATASET}"
echo "  DP policy : ${DP_POLICY_PATH}"
echo "  skillset  : ${SKILLSET_DIR}"
echo "  slurm     : partition=${SKILLVLA_PARTITION} qos=${SKILLVLA_QOS} gres=${SKILLVLA_GRES}"
echo "  array     : ${ARRAY_SPEC}  (${N_TASKS} tasks / ${TPJ} per job = ${NUM_SHARDS} GPUs)"

if [ "${SKILLSET_BOUNDARY_THRESHOLD_MODE}" = "episode_mean" ]; then
  JID_SEG=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" ALL_TASK_IDS="${ALL_TASK_IDS}" \
    sbatch --parsable --array="${ARRAY_SPEC}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
else
  JID_CURVES=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" ALL_TASK_IDS="${ALL_TASK_IDS}" CURVES_ONLY=true \
    sbatch --parsable --array="${ARRAY_SPEC}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  echo "  curve job    : ${JID_CURVES}"
  if [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ]; then
    echo "  threshold    : PT reference ${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
    JID_SEG=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" ALL_TASK_IDS="${ALL_TASK_IDS}" USE_CACHED_CURVES=true \
      sbatch --parsable --array="${ARRAY_SPEC}" --dependency="afterok:${JID_CURVES}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  else
    JID_REDUCE=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" EXPECTED_EPISODES="${EXPECTED_EPISODES}" \
      sbatch --parsable --dependency="afterok:${JID_CURVES}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/compute_global_boundary_threshold.sbatch")
    echo "  threshold job: ${JID_REDUCE}  (${EXPECTED_EPISODES} episodes)"
    JID_SEG=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" ALL_TASK_IDS="${ALL_TASK_IDS}" USE_CACHED_CURVES=true \
      sbatch --parsable --array="${ARRAY_SPEC}" --dependency="afterok:${JID_REDUCE}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  fi
fi
echo "  segment job  : ${JID_SEG}"

# verify + re-run any tasks left incomplete (e.g. a dead GPU); afterany → runs even
# if some array elements failed.
JIDV=$(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" \
  sbatch --parsable --dependency=afterany:"${JID_SEG}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/verify_skillset.sbatch")
echo "  verify job   : ${JIDV}  (after ${JID_SEG}; re-runs missing tasks up to ${SKILLSET_MAX_SWEEPS}×)"
