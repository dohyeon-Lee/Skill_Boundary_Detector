#!/usr/bin/env bash
# Step 5 (final): assemble the SkillVLA dataset and final outputs.
#   (b) add_skill_latents_to_dataset → skillvla/ (skill columns) + skill_initial_state.npz (ISS)
#   (c) copy FSQ checkpoint → FSQ.pt
#   (d) optionally build start/end skill_focus_uv.npz from recorded camera calibration
#   (e) optional cleanup of intermediates (_work, skill_latents.npz)
# DINO precompute is retired: terminator co-training tokenizes current batch
# frames online, without a per-frame observation.dino.image training cache.
#SBATCH --job-name=skillVLA_build
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # train_skillVLA
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SKILLVLA_ENV_SNAPSHOT:-}" ] && [ -f "${SKILLVLA_ENV_SNAPSHOT}" ]; then
  source "${SKILLVLA_ENV_SNAPSHOT}"
elif [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Fail over to another GPU instead of silently running on CPU.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA BUILD"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "Source   : ${RAW_DATASET_DIR}"
echo "Latents  : ${SKILL_LATENTS_PATH}"
echo "ISS npz  : ${ISS_NPZ_PATH}   (storage pmax=${SKILL_PMAX}, max_percent=${SKILL_JITTER_MAX_PERCENT}, gripper_validity=${SKILL_JITTER_PRECOMPUTE_GRIPPER_VALIDITY}, distribution=${SKILL_JITTER_DISTRIBUTION})"
echo "FSQ.pt   : ${FSQ_COPY_PATH}"
echo "skillvla : ${SKILLVLA_DATASET_DIR}"
echo "proprio  : grounding=${PROPRIO_GROUNDING}"
echo "outliers : exclude_rotation=${EXCLUDE_ROTATION_OUTLIER_EPISODES} threshold=${ROTATION_OUTLIER_THRESHOLD}"
echo "focus UV : enabled=${FOCUS_UV_ENABLED} camera=${FOCUS_UV_CAMERA} output=${FOCUS_UV_PATH}"

mkdir -p "${SKILLVLA_RUN_DIR}"

# (DINO precompute 단계 (a)/(a') 은퇴 — dino.npz/dino_wrist.npz 산출물 없음.)

# ── (b) add skill columns → SkillVLA LeRobot dataset + skill_initial_state.npz ──
BASE_COMPLETE=false
if [ -f "${ISS_NPZ_PATH}" ] && [ -f "${FSQ_COPY_PATH}" ] \
   && [ -d "${SKILLVLA_DATASET_DIR}" ] && [ -n "$(ls -A "${SKILLVLA_DATASET_DIR}" 2>/dev/null)" ] \
   && "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/check_jitter_contract.py" \
      --dataset-dir "${SKILLVLA_DATASET_DIR}" \
      --storage-pmax "${SKILL_PMAX}" \
      --max-percent "${SKILL_JITTER_MAX_PERCENT}" \
      --gripper-validity "${SKILL_JITTER_PRECOMPUTE_GRIPPER_VALIDITY}" \
      --distribution "${SKILL_JITTER_DISTRIBUTION}" >/dev/null; then
  BASE_COMPLETE=true
fi

if [ "${BASE_COMPLETE}" != "true" ]; then
  if [ ! -f "${SKILL_LATENTS_PATH}" ]; then
    echo "skill_latents.npz not found: ${SKILL_LATENTS_PATH}" >&2
    exit 1
  fi
  read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/configs/train_skillVLA/build_data/src/add_skill_latents_to_dataset.py \
      --src_dataset_dir "${RAW_DATASET_DIR}" \
      --dst_dataset_dir "${SKILLVLA_DATASET_DIR}" \
      --dst_repo_id     "skillvla/${SOURCE_DATASET}_${RUN_TAG}" \
      --latents_path    "${SKILL_LATENTS_PATH}" \
      --iss_npz_path    "${ISS_NPZ_PATH}" \
      --pmax            "${SKILL_PMAX}" \
      --jitter_max_percent "${SKILL_JITTER_MAX_PERCENT}" \
      --jitter_precompute_gripper_validity "${SKILL_JITTER_PRECOMPUTE_GRIPPER_VALIDITY}" \
      --jitter_distribution "${SKILL_JITTER_DISTRIBUTION}" \
      --proprio_grounding "${PROPRIO_GROUNDING}" \
      --rotation_outlier_threshold "${ROTATION_OUTLIER_THRESHOLD}" \
      --fsq_levels      "${FSQ_LEVELS[@]}" \
      --max_order       "${MAX_ORDER}" \
      --max_length      "${MAX_LENGTH}"

  # ── (c) copy the FSQ checkpoint used ──
  cp -f "${FSQ_MODEL_PATH}" "${FSQ_COPY_PATH}"
  echo "Copied FSQ checkpoint → ${FSQ_COPY_PATH}"
  "${PROJECT_ROOT}/.venv/bin/python" - \
    "${SKILLVLA_RUN_DIR}/fsq_source.json" \
    "${FSQ_RUN_NAME}" "${FSQ_CHECKPOINT}" \
    "${FSQ_MODEL_PATH}" "${FSQ_COPY_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
checkpoint = "best" if sys.argv[3] == "0" else sys.argv[3]
payload = {
    "schema_version": 1,
    "source_fsq_run_name": sys.argv[2],
    "source_fsq_checkpoint": checkpoint,
    "source_fsq_model_path": sys.argv[4],
    "copied_fsq_path": sys.argv[5],
}
temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
PY
  echo "Recorded FSQ provenance → ${SKILLVLA_RUN_DIR}/fsq_source.json"
else
  echo "Base SkillVLA dataset already complete; preserving it."
fi

# ── (d) canonical skill-start/end focus targets ──
if [ "${FOCUS_UV_ENABLED}" = "true" ]; then
  if ! python "${SRC_DIR}/build_skill_focus_uv.py" --check-current "${FOCUS_UV_PATH}"; then
    PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
      python -u "${SRC_DIR}/build_skill_focus_uv.py" \
        --skill-dataset-dir "${SKILLVLA_DATASET_DIR}" \
        --skill-latents-path "${SKILL_LATENTS_PATH}" \
        --eval-init-states-path "${FOCUS_UV_EVAL_INIT_STATES_PATH}" \
        --original-dataset-dir "${FOCUS_UV_ORIGINAL_DATASET_DIR}" \
        --suite "${FOCUS_UV_SUITE}" \
        --camera "${FOCUS_UV_CAMERA}" \
        --output "${FOCUS_UV_PATH}"
  else
    echo "Current start/end focus targets already exist → ${FOCUS_UV_PATH}"
  fi
else
  echo "focus_uv disabled → skipping camera UV targets; XYZ/state targets use parquet boundaries."
fi

# ── (e) cleanup intermediates ──
# Only this DP's segmentation (skillset + skill_tokens) and this run's latents.
# The shared dino/ stays so other DP/FSQ runs on the same source can reuse it.
if [ "${CLEANUP_INTERMEDIATE}" = "true" ]; then
  echo "Cleaning up intermediates ..."
  rm -rf "${SKILLVLA_SEG_DIR}"
  rm -f "${SKILL_LATENTS_PATH}"
  echo "  removed ${SKILLVLA_SEG_DIR} and ${SKILL_LATENTS_PATH}"
fi

echo "DONE → ${SKILLVLA_RUN_DIR}  (skillvla/, skill_initial_state.npz, FSQ.pt; focus_uv=${FOCUS_UV_ENABLED})"
