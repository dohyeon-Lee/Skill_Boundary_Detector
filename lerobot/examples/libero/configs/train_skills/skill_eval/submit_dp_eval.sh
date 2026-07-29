#!/usr/bin/env bash
# Inputs:
#   config : ./dp_eval_config.yaml  (roots + DP selection + HTML knobs + slurm)
# Outputs:
#   DP : ./outputs/dp_skillset/{dataset}/{dp_tag}_ck{ckpt}{suffix}.html
#
# Submit the DP skill-boundary eval (boxed start/end frames per skill + the
# multimodality curve). FSQ-independent; use submit_fsq_eval.sh for the FSQ eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}"

# This script runs exactly the DP eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_DP=true
export EVAL_RUN_FSQ=false

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
_probe="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" ${TARGET_DATASET:+--dataset "${TARGET_DATASET}"} --shell 2>/dev/null \
  | grep -E '^export (TARGET_DATASET|SKILLSET_DIR|SKILLSET_DONE_PATH|SKILLSET_MODE|SKILLSET_MIN_SKILLS|SKILLSET_OUTPUT_SUFFIX|SKILLSET_GLOBAL_THRESHOLD_SOURCE)=')"
eval "${_probe}"
NEED_BUILD=false
# Use the .complete marker, not just the skills dir — a PARTIAL skillset (some array shards failed) has
# a skills dir but no marker, so we still (re)build to finish it (build_data --resume skips done shards).
if [ -z "${DP_EVAL_SKILLSET_DIR:-}" ] && [ ! -f "${SKILLSET_DONE_PATH}" ]; then
  NEED_BUILD=true
fi
if [ "${NEED_BUILD}" = "true" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "DP eval skillset missing, but running under an existing allocation (srun) — cannot chain a" >&2
    echo "build_data job. Run build_data/submit_build_data.sh first (or submit eval without salloc)." >&2
    exit 1
  fi
  echo "DP eval skillset not built yet → submitting build_data first (skillset only)"
  echo "  DP     : ${EVAL_DP_RUN_NAME:-<build_data_config default>}  (skillset MISSING)"
  echo "  target : ${TARGET_DATASET}"
  # Build FOR THE EVAL's resolved target/DP/skillset selection — build_data reads its own
  # build_data_config.yaml only for build-time knobs (gripper mode, nms, ...). The dataset is
  # handed over as TRAIN_DATA (submit_build_data.sh's dataset input; it overwrites its own
  # TARGET_DATASET from it). Without this, build_data would build its yaml's own target, which
  # the eval's dependency wait would never see when the two configs point at different datasets.
  BUILD_OUT=$(TRAIN_DATA="${TARGET_DATASET}" \
                  DP_RUN_NAME="${EVAL_DP_RUN_NAME}" DP_CHECKPOINT="${EVAL_DP_CHECKPOINT}" \
                  SKILLSET_MODE="${SKILLSET_MODE}" SKILLSET_MIN_SKILLS="${SKILLSET_MIN_SKILLS}" \
                  SKILLSET_BOUNDARY_THRESHOLD_MODE="${SKILLSET_BOUNDARY_THRESHOLD_MODE}" \
                  SKILLSET_BOUNDARY_THRESHOLD_SCALE="${SKILLSET_BOUNDARY_THRESHOLD_SCALE}" \
                  SKILLSET_OUTPUT_SUFFIX="${SKILLSET_OUTPUT_SUFFIX}" \
                  SKILLSET_GLOBAL_THRESHOLD_SOURCE="${SKILLSET_GLOBAL_THRESHOLD_SOURCE}" \
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
  --job-name=dp_eval
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

echo "Submit DP skill-boundary eval"
echo "  DP          : ${DP_RUN_NAME} (ckpt ${DP_CHECKPOINT})"
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
