#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# ABC FSQ evaluation: checkpoint encoding followed by the CPU video report.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
# Optional --JOB_ID reuses that Slurm allocation.
source "$(dirname "${BASH_SOURCE[0]}")/../../../src/submit_args.sh"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
SHARED_DIR="${PROJECT_ROOT}/lerobot/skillVLA/libero/configs/train_skills/skill_eval/fsq_codebook"

export SBD_GLOBAL_CONFIG="${SCRIPT_DIR}/../../../global_config.yaml"
export FSQ_GT_REPLAY_DOMAIN=mimicgen
export FSQ_GT_REPLAY_DIR="${SCRIPT_DIR}"
export FSQ_GT_REPLAY_SRC_DIR="${SHARED_DIR}/src"
CONFIG="${MIMICGEN_FSQ_EVAL_CONFIG:-${FSQ_GT_REPLAY_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}}"
DRY_RUN=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG="${2:?--config requires a YAML path}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Usage: $0 [--JOB_ID] [--config FILE] [--dry-run]" >&2; exit 2 ;;
  esac
done
source "${SCRIPT_DIR}/../../../src/snapshot_config.sh"
if [ "${DRY_RUN}" != true ]; then
  CONFIG="$(snapshot_config "${CONFIG}")"
fi
export FSQ_GT_REPLAY_CONFIG="${CONFIG}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SETTINGS="$(env -u EVAL_MAX_CONCURRENT "${PYTHON}" "${SHARED_DIR}/src/config.py" --config "${CONFIG}" --shell)"
eval "${SETTINGS}"
echo "FSQ: ${FSQ_RUN_NAME} / ${FSQ_EVAL_CHECKPOINTS}"
echo "Report: ${EVAL_COLLECTION_DIR}/index.html"
echo "GPU encoding: ${FSQ_MISSING_LATENTS_COUNT} checkpoint(s); video report: CPU"
[ "${DRY_RUN}" != true ] || exit 0
if [ "${FSQ_RUN_COUNT}" -eq 1 ] && [ "${FSQ_PENDING_CHECKPOINT_COUNT}" -eq 0 ]; then
  echo "Already complete; no job submitted."
  exit 0
fi

# Queue the complete pipeline inside the selected allocation. The encoding and
# replay phases run in sequence there, without unsupported Slurm dependencies.
source "${SCRIPT_DIR}/../../../src/submit_job.sh"
if [ "$(submit_job_slurm_mode)" = srun_existing ]; then
  mkdir -p "${SCRIPT_DIR}/logs"
  cd "${SCRIPT_DIR}"
  ARGS=(--job-name=MimicGen_FSQ_eval --cpus-per-task="${EVAL_CPUS_PER_TASK}"
    --mem="${EVAL_MEM}" --time="${EVAL_LATENTS_TIME}"
    --output=logs/%x_%j.out --error=logs/%x_%j.err)
  [ "${FSQ_MISSING_LATENTS_COUNT}" -eq 0 ] || ARGS+=(--gres="${EVAL_GRES}")
  submit_job "${ARGS[@]}" "${SCRIPT_DIR}/src/run_eval.sbatch"
  exit 0
fi

exec bash "${SHARED_DIR}/submit.sh"
