#!/usr/bin/env bash
#SBATCH --job-name=ABC_skillVLA_build
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${ABC_SKILLVLA_CONFIG:-${SUBMIT_DIR}/train_skillVLA_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$(${BOOTSTRAP_PYTHON} "${SUBMIT_DIR}/src/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"

[ -f "${SKILL_LATENTS_PATH}" ] || { echo "Skill latents missing: ${SKILL_LATENTS_PATH}" >&2; exit 1; }
[ -f "${FSQ_MODEL_PATH}" ] || { echo "FSQ checkpoint missing: ${FSQ_MODEL_PATH}" >&2; exit 1; }
read -r -a FSQ_LEVELS <<< "${FSQ_LEVELS_STR}"
read -r -a SOURCE_TASKS <<< "${ABC_SOURCE_TASKS_STR}"
GRIPPER_ARG=()
[ "${TRANSITION_JITTER_PRECOMPUTE_GRIPPER_VALIDITY}" = true ] \
  && GRIPPER_ARG=(--precompute-gripper-validity)

echo "ABC native SkillVLA assembly"
echo "  source   : ${RAW_DATASET_DIR} (${ABC_SOURCE_TASKS_STR})"
echo "  latents  : ${SKILL_LATENTS_PATH}"
echo "  jitter   : pmax=${TRANSITION_JITTER_PMAX}, ${TRANSITION_JITTER_DISTRIBUTION}"
echo "  output   : ${SKILLVLA_RUN_DIR}"
CORE_COMPLETE=false
if [ -f "${SKILLVLA_RUN_DIR}/build_manifest.json" ] \
   && [ -f "${ISS_NPZ_PATH}" ] \
   && [ -f "${SKILLVLA_DATASET_DIR}/skill_annotations.npz" ]; then
  CORE_COMPLETE=true
  echo "  reuse    : existing annotations + ISS + manifest"
fi

if [ "${CORE_COMPLETE}" != true ]; then
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/abc/configs/train_skillVLA/build_data/src/build_abc_native_skillvla.py \
    --source-dataset-dir "${RAW_DATASET_DIR}" \
    --output-dataset-dir "${SKILLVLA_DATASET_DIR}" \
    --skill-latents-path "${SKILL_LATENTS_PATH}" \
    --iss-path "${ISS_NPZ_PATH}" \
    --stats-path "${FSQ_STATS_PATH}" \
    --repo-id "skillvla/${SOURCE_DATASET}_${SKILLVLA_RUN_NAME}" \
    --source-tasks "${SOURCE_TASKS[@]}" \
    --fsq-levels "${FSQ_LEVELS[@]}" \
    --max-order "${MAX_ORDER}" \
    --max-length "${MAX_LENGTH}" \
    --transition-jitter-pmax "${TRANSITION_JITTER_PMAX}" \
    --transition-jitter-distribution "${TRANSITION_JITTER_DISTRIBUTION}" \
      "${GRIPPER_ARG[@]}"
fi

if [ ! -f "${FSQ_COPY_PATH}" ]; then
  cp -f "${FSQ_MODEL_PATH}" "${FSQ_COPY_PATH}"
fi
python - <<PY
import json
from pathlib import Path
Path("${SKILLVLA_RUN_DIR}/fsq_source.json").write_text(json.dumps({
    "schema_version": 1,
    "source_fsq_run_name": "${FSQ_RUN_NAME}",
    "source_fsq_checkpoint": "${FSQ_CHECKPOINT}",
    "source_fsq_model_path": "${FSQ_MODEL_PATH}",
    "copied_fsq_path": "${FSQ_COPY_PATH}",
}, indent=2) + "\\n")
PY

if [ "${SPATIAL_TARGETS_ENABLED}" = true ]; then
  KEEP_SHARDS_ARG=()
  [ "${SPATIAL_KEEP_SHARDS}" = true ] && KEEP_SHARDS_ARG=(--keep-shards)
  echo "ABC exact bimanual spatial targets"
  echo "  output   : ${SPATIAL_TARGETS_PATH}"
  echo "  workers  : ${SPATIAL_WORKERS}"
  python -u examples/abc/configs/train_skillVLA/build_data/src/build_abc_spatial_targets.py \
    --skill-dataset-dir "${SKILLVLA_DATASET_DIR}" \
    --skill-latents-path "${SKILL_LATENTS_PATH}" \
    --annotations-path "${SKILL_ANNOTATIONS_PATH}" \
    --assets-dir "${SPATIAL_ASSETS_DIR}" \
    --output "${SPATIAL_TARGETS_PATH}" \
    --shard-dir "${SPATIAL_SHARD_DIR}" \
    --workers "${SPATIAL_WORKERS}" \
    "${KEEP_SHARDS_ARG[@]}"
fi

if [ "${CLEANUP_INTERMEDIATE}" = true ]; then
  rm -f "${SKILL_LATENTS_PATH}"
fi
echo "DONE -> ${SKILLVLA_RUN_DIR}"
