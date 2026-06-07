#!/usr/bin/env bash
# Run the full SkillVLA data-generation pipeline in one submission:
#   (login)  slice per-episode DINO from the base full DINO
#   job 1    build_skillset.sbatch  — DP skill segmentation (Slurm array over tasks)
#   job 1b   verify_skillset.sbatch — verify + re-run tasks a dead GPU missed (afterany:1)
#   job 2    encode_skills.sbatch   — extract skill DINO tokens + FSQ encode   (after 1b)
#   job 3    build_skillvla.sbatch  — dino.npz + skillvla/ + skill_initial_state.npz + FSQ.pt  (after 2)
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
encode_complete () { [ -f "${SKILL_TOKENS_PATH}" ] && [ -f "${SKILL_LATENTS_PATH}" ]; }

# ── step 2 (login): slice per-episode DINO (idempotent — skips existing) ──
echo "[1/4] Slice per-episode DINO  ${BASE_DINO_DIR} → ${DINO_PER_EPISODE_DIR}"
"${BOOTSTRAP_PYTHON}" "${SRC_DIR}/prepare_dino_for_skillvla.py" \
  --dataset_dir   "${RAW_DATASET_DIR}" \
  --base_dino_dir "${BASE_DINO_DIR}" \
  --dst_dir       "${DINO_PER_EPISODE_DIR}" \
  --image_keys    "${DINO_IMAGE_KEYS}" \
  --mode          "${DINO_COPY_MODE}"

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

# ── stage 3: skillset (GPU array over tasks) + verify/re-run ──
if skillset_complete; then
  echo "[2/4] Skillset already complete → skip"
else
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
  [ "${SKILLSET_ARRAY_THROTTLE:-0}" -gt 0 ] && ARRAY_SPEC="${ARRAY_SPEC}%${SKILLSET_ARRAY_THROTTLE}"

  echo "[2/4] Submit DP skill segmentation  (array=${ARRAY_SPEC}: ${N_TASKS} tasks / ${TPJ} per job = ${NUM_SHARDS} GPUs)"
  JID1=$(env "${ENV[@]}" ALL_TASK_IDS="${ALL_TASK_IDS}" \
    sbatch --parsable --array="${ARRAY_SPEC}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillset.sbatch")
  echo "       skillset ${JID1}"
  # afterany → verify runs even if some array elements died on a bad GPU
  JIDV=$(env "${ENV[@]}" \
    sbatch --parsable --dependency=afterany:"${JID1}" "${SBATCH_ARGS[@]}" "${SRC_DIR}/verify_skillset.sbatch")
  echo "       verify   ${JIDV}  (re-runs missing tasks up to ${SKILLSET_MAX_SWEEPS}×)"
  DEP="afterok:${JIDV}"
fi

# ── stage 4: FSQ encode ──
if encode_complete; then
  echo "[3/4] Skill tokens + latents already exist → skip"
else
  DEP_ARG=(); [ -n "${DEP}" ] && DEP_ARG=(--dependency="${DEP}")
  echo "[3/4] Submit FSQ encode${DEP:+  (after ${DEP#afterok:})}"
  JID2=$(env "${ENV[@]}" \
    sbatch --parsable ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SBATCH_ARGS[@]}" "${SRC_DIR}/encode_skills.sbatch")
  echo "       encode ${JID2}"
  DEP="afterok:${JID2}"
fi

# ── stage 5: SkillVLA build (always — would have exited above if already done) ──
DEP_ARG=(); [ -n "${DEP}" ] && DEP_ARG=(--dependency="${DEP}")
echo "[4/4] Submit SkillVLA build${DEP:+  (after ${DEP#afterok:})}"
JID3=$(env "${ENV[@]}" \
  sbatch --parsable ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${SBATCH_ARGS[@]}" "${SRC_DIR}/build_skillvla.sbatch")
echo "       build ${JID3}"

echo "Submitted. Final outputs → ${SKILLVLA_RUN_DIR}  (skillvla/, skill_initial_state.npz, dino.npz, FSQ.pt)"
