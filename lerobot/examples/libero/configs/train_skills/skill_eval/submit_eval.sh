#!/usr/bin/env bash
# Inputs:
#   roots        : ../train_skills_config.yaml  (skills / DINO / dataset / outputs/FSQ root)
#   eval knobs   : ./eval_config.yaml           (eval_run_fsq/eval_run_dp, run_name, checkpoint, slurm)
#   FSQ model    : {project_root}/outputs/FSQ/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch*.pt)
# Outputs (per eval_run_fsq / eval_run_dp):
#   FSQ : ./outputs/{fsq_eval_run_name}/{epoch}/fsq_eval.html
#   DP  : ./outputs/dp_skillset/{dataset}/index.html
#
# Submit FSQ reconstruction eval and/or the DP skill-boundary eval (toggled in eval_config.yaml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/eval_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${TRAIN_CONFIG}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
TRAIN_CONFIG="$(snapshot_config "${TRAIN_CONFIG}")"
EVAL_CONFIG="$(snapshot_config "${EVAL_CONFIG}")"
TARGET_DATASET="${TRAIN_DATA:-}"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Evaluation-only knobs + slurm — sourced FIRST so DP-selection overrides can steer the shared resolver.
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/eval_config.py" --config "${EVAL_CONFIG}" --shell)"
# Optionally evaluate a DIFFERENT DP than the shared config points at (give the DP folder name).
[ -n "${EVAL_DP_RUN_NAME}" ]   && export DP_RUN_NAME="${EVAL_DP_RUN_NAME}"
[ -n "${EVAL_DP_CHECKPOINT}" ] && export DP_CHECKPOINT="${EVAL_DP_CHECKPOINT}"
# ── Auto-build (sbatch mode): if the skillset this eval needs is not built yet, submit build_data
# first (for the eval's chosen DP) and make the eval wait on it. CRITICAL: we resolve ONLY
# the few paths checked here (in a subshell) and do NOT export the eval's full config before calling
# build_data, so evaluation-only overrides cannot leak through the environment and replace the
# build-data configuration (get_value prefers env over yaml).
DEP_ARG=()
# DP eval needs eval_dp_run_name's skillset; FSQ eval reads its OWN skillset (recorded in fsq_meta.json),
# which already exists since the FSQ trained on it. So only the DP eval can require an auto-build.
_probe="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" ${TARGET_DATASET:+--dataset "${TARGET_DATASET}"} --shell 2>/dev/null \
  | grep -E '^export (SKILLSET_DIR|SKILLSET_DONE_PATH)=')"
eval "${_probe}"
NEED_BUILD=false
# Use the .complete marker, not just the skills dir — a PARTIAL skillset (some array shards failed) has
# a skills dir but no marker, so we still (re)build to finish it (build_data --resume skips done shards).
if [ "${EVAL_RUN_DP}" = "true" ] && [ -z "${DP_EVAL_SKILLSET_DIR:-}" ] && \
   [ ! -f "${SKILLSET_DONE_PATH}" ]; then
  NEED_BUILD=true
fi
if [ "${NEED_BUILD}" = "true" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "DP eval skillset missing, but running under an existing allocation (srun) — cannot chain a" >&2
    echo "build_data job. Run build_data/submit_build_data.sh first (or submit eval without salloc)." >&2
    exit 1
  fi
  echo "DP eval skillset not built yet → submitting build_data first (skillset only)"
  echo "  DP   : ${EVAL_DP_RUN_NAME:-<build_data_config default>}  (skillset MISSING)"
  # Strip eval-side config vars so build_data reads its own build_data_config.yaml; pass only the DP
  # target. BUILD_SKILLSET_ONLY=true preserves the caller's intent; current build_data produces only
  # the skillset. FSQ evaluation decodes selected raw frames live from its recorded skillset.
  BUILD_OUT=$(env -u TARGET_DATASET \
                  DP_RUN_NAME="${EVAL_DP_RUN_NAME}" DP_CHECKPOINT="${EVAL_DP_CHECKPOINT}" \
                  SKILLSET_BOUNDARY_THRESHOLD_MODE="${SKILLSET_BOUNDARY_THRESHOLD_MODE}" \
                  BUILD_SKILLSET_ONLY=true PRINT_LAST_JOB=1 \
    bash "${SCRIPT_DIR}/../build_data/submit_build_data.sh") || { echo "build_data submission failed" >&2; exit 1; }
  echo "${BUILD_OUT}"
  BUILD_JOB=$(echo "${BUILD_OUT}" | grep -oE "LAST_JOB=[0-9]+" | tail -1 | cut -d= -f2)
  if [ -z "${BUILD_JOB}" ]; then
    echo "Could not capture build_data job id — aborting." >&2; exit 1
  fi
  DEP_ARG=(--dependency=afterok:"${BUILD_JOB}")
  echo "  eval will run after build_data job ${BUILD_JOB}"
fi

# NOW source the full eval config for the submission below (SBATCH_ARGS + the eval job). Any env
# "pollution" here is harmless: build_data already ran, and the eval job re-resolves from TRAIN_CONFIG.
if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TARGET_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi

SBATCH_ARGS=(
  --partition="${FSQ_EVAL_PARTITION}"
  --qos="${FSQ_EVAL_QOS}"
  --gres="${FSQ_EVAL_GRES}"
  --cpus-per-task="${FSQ_EVAL_CPUS_PER_TASK}"
  --mem="${FSQ_EVAL_MEM}"
  --time="${FSQ_EVAL_TIME}"
)
if [ -n "${FSQ_EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_EVAL_NODELIST}")
fi
if [ -n "${FSQ_EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs outputs

echo "Submit DP_FSQ eval"
echo "  run_fsq     : ${EVAL_RUN_FSQ}   run_dp : ${EVAL_RUN_DP}"
echo "  FSQ run     : ${FSQ_EVAL_RUN_NAME} (ckpt ${FSQ_EVAL_CHECKPOINT})"
echo "  skills      : ${SKILLSET_DIR}/skills"
echo "  curves      : ${SKILLSET_DIR}/curves"
echo "  dataset     : ${DATASET_ROOT}/${TARGET_DATASET}"
echo "  slurm       : partition=${FSQ_EVAL_PARTITION} qos=${FSQ_EVAL_QOS} gres=${FSQ_EVAL_GRES}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  echo "  mode        : srun (reusing allocation ${SLURM_JOB_ID})"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" TRAIN_SKILLS_CONFIG="${TRAIN_CONFIG}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" TRAIN_DATA="${TARGET_DATASET}" \
    srun "${EVAL_SRC_DIR}/eval.sbatch"
else
  echo "  mode        : sbatch (new job)${DEP_ARG[*]:+  ${DEP_ARG[*]}}"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" TRAIN_SKILLS_CONFIG="${TRAIN_CONFIG}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" TRAIN_DATA="${TARGET_DATASET}" \
    sbatch "${SBATCH_ARGS[@]}" ${DEP_ARG[@]+"${DEP_ARG[@]}"} "${EVAL_SRC_DIR}/eval.sbatch"
fi
