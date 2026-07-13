#!/usr/bin/env bash
# Backfill the per-episode multimodality (VF cos-divergence) curves for an ALREADY-BUILT
# skillset, so the eval HTML (run_skillset) can overlay the boundary-criteria graph.
#
# Why a dedicated script (not `sbatch build_skillset.sbatch` directly):
#   • build_skillset.sbatch has NO #SBATCH --partition/--qos/--nodelist; those are injected
#     at submit time from global_config (SKILLVLA_*). A bare sbatch lands on the default
#     partition (wrong GPU).
#   • the sbatch anchors paths on SLURM_SUBMIT_DIR and expects CWD = the build_data dir
#     (this script cd's there); a bare sbatch from src/ doubles SRC_DIR → src/src and the
#     config sourcing fails (LEROBOT_ROOT unbound).
#   • submit_build_skillset.sh early-exits when the skillset is already complete, so it
#     would refuse to run for a finished run. This script skips that check.
#
# It reuses the existing skills/DINO/DP — runs DP only to recompute div_cos, writes
# {SKILLSET_DIR}/curves/ep*.npz (resume keyed by the curve file), and does NOT touch
# skills or .done markers (build_skill_dataset --curves_only).
#
# Usage (from anywhere):
#   .../build_data/src/submit_backfill_curves.sh
#   SOURCE_DATA=libero_90_full_10 .../submit_backfill_curves.sh   # override source dataset
#   TRAIN_SKILLVLA_CONFIG=/path/to/train_skillVLA_config.yaml .../submit_backfill_curves.sh

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
  exit 1
fi

# ── task-shard array (one shard = SKILLSET_TASKS_PER_JOB tasks → 1 GPU) ──
ALL_TASK_IDS="$("${BOOTSTRAP_PYTHON}" - "${RAW_DATASET_DIR}/meta/tasks.parquet" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1]).reset_index()
print(" ".join(str(int(t)) for t in sorted(df["task_index"].unique())))
PY
)"
N_TASKS=$(wc -w <<< "${ALL_TASK_IDS}")
TPJ="${SKILLSET_TASKS_PER_JOB:-5}"
NUM_SHARDS=$(( (N_TASKS + TPJ - 1) / TPJ ))
[ "${NUM_SHARDS}" -lt 1 ] && NUM_SHARDS=1
ARRAY_SPEC="0-$(( NUM_SHARDS - 1 ))"
if [ "${SKILLSET_ARRAY_THROTTLE:-0}" -gt 0 ]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${SKILLSET_ARRAY_THROTTLE}"
fi

# ── slurm args from global_config (the part a bare sbatch was missing) ──
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

# cd into the build_data dir so the sbatch's SLURM_SUBMIT_DIR anchor resolves (SRC_DIR=$SUBMIT_DIR/src).
cd "${SCRIPT_DIR}/.."
mkdir -p logs

echo "Backfill multimodality curves (CURVES_ONLY)"
echo "  source    : ${SOURCE_DATASET:-<default>}"
echo "  DP policy : ${DP_POLICY_PATH}"
echo "  curves    : ${SKILLSET_DIR}/curves"
echo "  slurm     : partition=${SKILLVLA_PARTITION} qos=${SKILLVLA_QOS} gres=${SKILLVLA_GRES}"
echo "  array     : ${ARRAY_SPEC}  (${N_TASKS} tasks / ${TPJ} per job = ${NUM_SHARDS} GPUs)"

JID=$(CURVES_ONLY=true TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" \
  ALL_TASK_IDS="${ALL_TASK_IDS}" \
  sbatch --parsable --array="${ARRAY_SPEC}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
echo "  curves backfill job : ${JID}"
echo "When done, re-run the eval (run_skillset) — the graph reads ${SKILLSET_DIR}/curves."
