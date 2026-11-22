#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
# ABC native SkillVLA data pipeline:
#   1) encode the already-built FSQ training skillset with the selected epoch
#   2) build storage-light native annotations + ISS and copy FSQ.pt
set -euo pipefail
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../src/submit_args.sh"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ABC_SKILLVLA_CONFIG:-${SCRIPT_DIR}/train_skillVLA_config.yaml}"

_lib="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
if ! EXPORTS="$(${BOOTSTRAP_PYTHON} "${SCRIPT_DIR}/src/train_skillVLA_config.py" \
  --config "${CONFIG_PATH}" --shell)"; then
  echo "ABC SkillVLA build configuration failed; no job was submitted." >&2
  exit 1
fi
eval "${EXPORTS}"

[ -d "${SKILLSET_DIR}/skills" ] || { echo "Skillset missing: ${SKILLSET_DIR}/skills" >&2; exit 1; }
[ -f "${FSQ_MODEL_PATH}" ] || {
  echo "FSQ checkpoint is not ready yet: ${FSQ_MODEL_PATH}" >&2
  echo "Run this command again after epoch ${FSQ_CHECKPOINT} is saved." >&2
  exit 1
}

if [ -f "${SKILLVLA_RUN_DIR}/build_manifest.json" ] \
   && [ -f "${ISS_NPZ_PATH}" ] \
   && [ -f "${FSQ_COPY_PATH}" ] \
   && [ -f "${SKILLVLA_DATASET_DIR}/skill_annotations.npz" ] \
   && { [ "${SPATIAL_TARGETS_ENABLED}" != true ] || [ -f "${SPATIAL_TARGETS_PATH}" ]; }; then
  echo "ABC SkillVLA dataset already complete: ${SKILLVLA_RUN_DIR}"
  exit 0
fi

mkdir -p "${SCRIPT_DIR}/logs"
# Existing allocations run both phases in one worker; batch dependency and
# parsable options cannot be used by the allocation runner.
if [ "$(submit_job_scheduler)" = slurm ] \
   && [ "$(submit_job_slurm_mode)" = srun_existing ]; then
  PIPELINE_CPUS="${SKILLVLA_BUILD_CPUS_PER_TASK}"
  if [ "${SKILLVLA_ENCODE_CPUS_PER_TASK}" -gt "${PIPELINE_CPUS}" ]; then
    PIPELINE_CPUS="${SKILLVLA_ENCODE_CPUS_PER_TASK}"
  fi
  cd "${SCRIPT_DIR}"
  echo "Queue ABC SkillVLA encode + build in the selected allocation"
  ABC_SKILLVLA_CONFIG="${CONFIG_PATH}" submit_job \
    --job-name=ABC_skillVLA_build \
    --gres="${SKILLVLA_ENCODE_GRES}" \
    --cpus-per-task="${PIPELINE_CPUS}" \
    --mem="${SKILLVLA_BUILD_MEM}" \
    --time="${SKILLVLA_BUILD_TIME}" \
    --output=logs/%x_%j.out --error=logs/%x_%j.err \
    "${SCRIPT_DIR}/src/build_existing.sbatch"
  echo "Final outputs -> ${SKILLVLA_RUN_DIR}"
  exit 0
fi

ENCODE_ARGS=(
  --partition="${SKILLVLA_ENCODE_PARTITION}"
  --qos="${SKILLVLA_ENCODE_QOS}"
  --gres="${SKILLVLA_ENCODE_GRES}"
  --cpus-per-task="${SKILLVLA_ENCODE_CPUS_PER_TASK}"
  --mem="${SKILLVLA_ENCODE_MEM}"
  --time="${SKILLVLA_ENCODE_TIME}"
)
[ -n "${SKILLVLA_ENCODE_NODELIST}" ] && ENCODE_ARGS+=(--nodelist="${SKILLVLA_ENCODE_NODELIST}")
[ -n "${SKILLVLA_ENCODE_EXCLUDE_NODES}" ] && ENCODE_ARGS+=(--exclude="${SKILLVLA_ENCODE_EXCLUDE_NODES}")

DEPENDENCY=()
if [ -f "${SKILL_LATENTS_PATH}" ]; then
  echo "[1/2] Encoded skill latents already exist -> skip"
else
  echo "[1/2] Submit FSQ epoch ${FSQ_CHECKPOINT} skill encoding"
  ENCODE_JOB=$(ABC_SKILLVLA_CONFIG="${CONFIG_PATH}" \
    submit_job --parsable "${ENCODE_ARGS[@]}" \
    "${SCRIPT_DIR}/src/encode_skills.sbatch")
  echo "      encode ${ENCODE_JOB}"
  DEPENDENCY=(--dependency="afterok:${ENCODE_JOB}" --kill-on-invalid-dep=yes)
fi

BUILD_ARGS=(
  --partition="${SKILLVLA_BUILD_PARTITION}"
  --qos="${SKILLVLA_BUILD_QOS}"
  --cpus-per-task="${SKILLVLA_BUILD_CPUS_PER_TASK}"
  --mem="${SKILLVLA_BUILD_MEM}"
  --time="${SKILLVLA_BUILD_TIME}"
)
[ -n "${SKILLVLA_BUILD_ACCOUNT}" ] && BUILD_ARGS+=(--account="${SKILLVLA_BUILD_ACCOUNT}")
[ -n "${SKILLVLA_BUILD_EXCLUDE_NODES}" ] && BUILD_ARGS+=(--exclude="${SKILLVLA_BUILD_EXCLUDE_NODES}")

echo "[2/2] Submit native annotation/ISS + spatial-target assembly"
BUILD_JOB=$(ABC_SKILLVLA_CONFIG="${CONFIG_PATH}" \
  submit_job --parsable "${BUILD_ARGS[@]}" \
  "${DEPENDENCY[@]}" "${SCRIPT_DIR}/src/build_skillvla.sbatch")
echo "      build  ${BUILD_JOB}"
echo "Final outputs -> ${SKILLVLA_RUN_DIR}"
