#!/usr/bin/env bash
# Run the full SkillVLA data-generation pipeline in one submission (DINO는 어디서도 precompute
# 안 함 — DP는 state/raw-frames, FSQ 학습·terminator는 ONLINE):
#   job 1    build_skillset.sbatch  — DP skill segmentation (Slurm array over tasks)
#   job 1b   verify_skillset.sbatch — verify + re-run tasks a dead GPU missed (afterany:1)
#   job 2    encode_skills.sbatch   — FSQ encode → skill_latents.npz            (after 1b)
#   job 3    build_skillvla.sbatch  — skillvla/ + skill_initial_state.npz + FSQ.pt  (after 2)
#
# Stages whose outputs already exist are skipped and the --dependency chain is
# rewired around them. Final outputs land in {skillvla_dataset}/{source}/{run_tag}/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SCRIPT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

mkdir -p "${SCRIPT_DIR}/logs"
SNAPSHOT_ENV="${SCRIPT_DIR}/logs/skillvla_env_$(date +%Y%m%d_%H%M%S)_$$.sh"
if [ -n "${SOURCE_DATASET}" ]; then
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell > "${SNAPSHOT_ENV}"
else
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell > "${SNAPSHOT_ENV}"
fi
# Freeze all downstream jobs to the exact config resolved at submit time.
source "${SNAPSHOT_ENV}"

# ── short-circuit: whole pipeline already done? (final outputs survive cleanup) ──
build_complete () {
  [ -f "${ISS_NPZ_PATH}" ] && [ -f "${FSQ_COPY_PATH}" ] \
    && [ -d "${SKILLVLA_DATASET_DIR}" ] && [ -n "$(ls -A "${SKILLVLA_DATASET_DIR}" 2>/dev/null)" ]
}
if build_complete; then
  echo "SkillVLA outputs already exist → nothing to do (${SKILLVLA_RUN_DIR})"
  exit 0
fi

# ── prerequisite checks ──
if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Source dataset not found: ${RAW_DATASET_DIR}" >&2; exit 1
fi
if [ ! -d "${DP_POLICY_PATH}" ]; then
  echo "DP policy not found: ${DP_POLICY_PATH}" >&2; exit 1
fi
if [ ! -f "${FSQ_MODEL_PATH}" ]; then
  echo "FSQ model not found: ${FSQ_MODEL_PATH}" >&2; exit 1
fi

# ── per-stage completeness checks (for skipping) ──
skillset_complete () {
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/verify_skillset.py" \
    --dataset_dir "${RAW_DATASET_DIR}" --skillset_dir "${SKILLSET_DIR}" --check >/dev/null 2>&1
}
encode_complete () { [ -f "${SKILL_LATENTS_PATH}" ]; }   # skill_tokens.npz 은퇴 (소비자 없음)

cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --partition="${SKILLVLA_PARTITION}" --qos="${SKILLVLA_QOS}" --gres="${SKILLVLA_GRES}"
  --cpus-per-task="${SKILLVLA_CPUS_PER_TASK}" --mem="${SKILLVLA_MEM}" --time="${SKILLVLA_TIME}"
)
[ -n "${SKILLVLA_NODELIST}" ]      && SBATCH_ARGS+=(--nodelist="${SKILLVLA_NODELIST}")
[ -n "${SKILLVLA_EXCLUDE_NODES}" ] && SBATCH_ARGS+=(--exclude="${SKILLVLA_EXCLUDE_NODES}")

ENV=(TRAIN_SKILLVLA_CONFIG="${CONFIG_PATH}" SOURCE_DATA="${SOURCE_DATASET}" SKILLVLA_ENV_SNAPSHOT="${SNAPSHOT_ENV}")
DEP=""   # dependency clause (e.g. "afterok:123") carried to the next non-skipped stage

# (구 step 2: per-episode DINO precompute 은퇴 — DINO는 어디서도 precompute 안 함. DP segmentation은
#  state/raw-frames, FSQ 학습·terminator는 ONLINE. 세그먼테이션이 곧 첫 스테이지.)

# ── stage: skillset (GPU array over tasks) + verify/re-run ──
if skillset_complete; then
  echo "[1/3] Skillset already complete → skip"
else
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
  [ "${SKILLSET_ARRAY_THROTTLE:-0}" -gt 0 ] && ARRAY_SPEC="${ARRAY_SPEC}%${SKILLSET_ARRAY_THROTTLE}"

  echo "[1/3] Submit global-boundary curve collection  (array=${ARRAY_SPEC}: ${N_TASKS} tasks / ${TPJ} per job = ${NUM_SHARDS} GPUs)"
  DEP_ARG=(); [ -n "${DEP}" ] && DEP_ARG=(--dependency="${DEP}" --kill-on-invalid-dep=yes)
  JID_CURVES=$(env "${ENV[@]}" ALL_TASK_IDS="${ALL_TASK_IDS}" CURVES_ONLY=true \
    sbatch --parsable --array="${ARRAY_SPEC}" ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  echo "       curves ${JID_CURVES}"
  if [ -n "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" ]; then
    echo "       threshold: PT reference ${SKILLSET_GLOBAL_THRESHOLD_SOURCE}"
    JID_SEG=$(env "${ENV[@]}" ALL_TASK_IDS="${ALL_TASK_IDS}" USE_CACHED_CURVES=true \
      sbatch --parsable --array="${ARRAY_SPEC}" --dependency="afterok:${JID_CURVES}" --kill-on-invalid-dep=yes "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  else
    JID_REDUCE=$(env "${ENV[@]}" EXPECTED_EPISODES="${EXPECTED_EPISODES}" \
      sbatch --parsable --dependency="afterok:${JID_CURVES}" --kill-on-invalid-dep=yes "${SBATCH_ARGS[@]}" "${SRC_DIR}/compute_global_boundary_threshold.sbatch")
    echo "       threshold ${JID_REDUCE}  (${EXPECTED_EPISODES} episodes)"
    JID_SEG=$(env "${ENV[@]}" ALL_TASK_IDS="${ALL_TASK_IDS}" USE_CACHED_CURVES=true \
      sbatch --parsable --array="${ARRAY_SPEC}" --dependency="afterok:${JID_REDUCE}" --kill-on-invalid-dep=yes "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  fi
  echo "       segment ${JID_SEG}"
  # afterany → verify runs even if some array elements died on a bad GPU
  JIDV=$(env "${ENV[@]}" \
    sbatch --parsable --dependency=afterany:"${JID_SEG}" --kill-on-invalid-dep=yes "${SBATCH_ARGS[@]}" "${SRC_DIR}/verify_skillset.sbatch")
  echo "       verify   ${JIDV}  (re-runs cached-curve segmentation gaps up to ${SKILLSET_MAX_SWEEPS}×)"
  DEP="afterok:${JIDV}"
fi

# ── stage 4: FSQ encode ──
if encode_complete; then
  echo "[2/3] Skill latents already exist → skip"
else
  DEP_ARG=(); [ -n "${DEP}" ] && DEP_ARG=(--dependency="${DEP}" --kill-on-invalid-dep=yes)
  echo "[2/3] Submit FSQ encode${DEP:+  (after ${DEP#afterok:})}"
  JID2=$(env "${ENV[@]}" \
    sbatch --parsable ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SBATCH_ARGS[@]}" "${SRC_DIR}/encode_skills.sbatch")
  echo "       encode ${JID2}"
  DEP="afterok:${JID2}"
fi

# ── stage 5: SkillVLA build (always — would have exited above if already done) ──
DEP_ARG=(); [ -n "${DEP}" ] && DEP_ARG=(--dependency="${DEP}" --kill-on-invalid-dep=yes)
echo "[3/3] Submit SkillVLA build${DEP:+  (after ${DEP#afterok:})}"
JID3=$(env "${ENV[@]}" \
  sbatch --parsable ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillvla.sbatch")
echo "       build ${JID3}"

echo "Submitted. Final outputs → ${SKILLVLA_RUN_DIR}  (skillvla/, skill_initial_state.npz, FSQ.pt)"
