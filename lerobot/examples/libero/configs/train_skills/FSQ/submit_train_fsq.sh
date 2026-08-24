#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in ../train_skills_config.yaml
#   skillset     : selected by the five folder components in fsq_config.yaml;
#                  DP/detector provenance is read from skillset_manifest.json
# Reference models:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   FSQ run      : {project_root}/outputs/FSQ/{fsq_run_name}
#   FSQ checkpoints: {project_root}/outputs/FSQ/{fsq_run_name}/FSQ_epoch*.pt
#   skill tokens : {project_root}/outputs/FSQ/{fsq_run_name}/skill_latents.npz
#
# Submit FSQ training using Slurm values from train_skills_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SRC_DIR="${SCRIPT_DIR}/../src"
FSQ_SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/fsq_config.yaml}"

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"
TARGET_DATASET="${TRAIN_DATA:-}"

# Config resolution has no project-runtime dependency; avoid waking the Lustre
# .venv before the actual training process needs it.
BOOTSTRAP_PYTHON=/usr/bin/python3

if [ -n "${TARGET_DATASET}" ]; then
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TARGET_DATASET}" --shell)"
else
  RESOLVED_SETTINGS="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "${RESOLVED_SETTINGS}"

# Build one immutable, compressed venv file on the submit host. Compute nodes
# then perform one sequential Lustre read instead of thousands of metadata
# lookups. This is scoped to FSQ and may be disabled with FSQ_NODE_LOCAL_VENV=0.
source "${FSQ_SRC_DIR}/fsq_node_local_venv.sh"
FSQ_VENV_ARCHIVE=""
if [ "${FSQ_NODE_LOCAL_VENV:-1}" = "1" ]; then
  if ! FSQ_VENV_ARCHIVE="$(fsq_prepare_venv_archive "${PROJECT_ROOT}")"; then
    FSQ_VENV_ARCHIVE=""
    echo "FSQ venv cache: preparation failed; submitted job will use the shared venv." >&2
  fi
fi

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "Skillset not found: ${SKILLSET_DIR}/skills" >&2
  echo "Run build_data/submit_build_data.sh first." >&2
  exit 1
fi
# Image/both terminators decode their sampled frames live. State-only and
# reconstructor-only jobs never touch the video directory.
USES_VISUAL_TERMINATOR=false
if { [ "${FSQ_DECODER_TERMINATOR_PROGRESS}" = "true" ] || [ "${FSQ_DECODER_TERMINATOR_TERMINATION}" = "true" ]; } \
  && [ "${FSQ_TERMINATOR_INPUT_SPACE}" != "state" ]; then
  USES_VISUAL_TERMINATOR=true
fi
if [ "${USES_VISUAL_TERMINATOR}" = "true" ] && [ ! -d "${RAW_DATASET_DIR}/videos" ]; then
  echo "Raw dataset videos not found: ${RAW_DATASET_DIR}/videos" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${FSQ_TRAIN_PARTITION}"
  --qos="${FSQ_TRAIN_QOS}"
  --gres="${FSQ_TRAIN_GRES}"
  --cpus-per-task="${FSQ_TRAIN_CPUS_PER_TASK}"
  --mem="${FSQ_TRAIN_MEM}"
  --time="${FSQ_TRAIN_TIME}"
)

if [ -n "${FSQ_TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_TRAIN_NODELIST}")
fi
if [ -n "${FSQ_TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_TRAIN_EXCLUDE_NODES}")
fi

# Resolve or submit the one shared lossless-zstd RGB producer before a GPU.
# flock serializes concurrent submit commands; the potentially large producer
# is CPU-only, and every visual training job attaches with afterok.
FRAME_CACHE_DEPENDENCY_ARGS=()
if [ "${USES_VISUAL_TERMINATOR}" = "true" ] && [ "${FSQ_FRAME_CACHE_ENABLED}" = "true" ]; then
  FRAME_CACHE_TOOL="${LEROBOT_ROOT}/examples/libero/fsq_frame_cache.py"
  mkdir -p "${FSQ_FRAME_CACHE_ROOT}/.jobs"
  exec 9>"${FSQ_FRAME_CACHE_ROOT}/.submit.lock"
  flock 9
  FRAME_CACHE_STATUS="$(${BOOTSTRAP_PYTHON} "${FRAME_CACHE_TOOL}" status \
    --raw-dataset-dir "${RAW_DATASET_DIR}" \
    --cache-root "${FSQ_FRAME_CACHE_ROOT}" \
    --shell)"
  eval "${FRAME_CACHE_STATUS}"

  if [ "${FSQ_FRAME_CACHE_COMPLETE}" != "true" ]; then
    FRAME_CACHE_JOB_ID=""
    if [ -f "${FSQ_FRAME_CACHE_JOB_FILE}" ]; then
      CANDIDATE_JOB_ID="$(tr -dc '0-9' < "${FSQ_FRAME_CACHE_JOB_FILE}")"
      if [ -n "${CANDIDATE_JOB_ID}" ]; then
        CANDIDATE_STATE="$(squeue -h -j "${CANDIDATE_JOB_ID}" -o '%T' 2>/dev/null | head -1 || true)"
        case "${CANDIDATE_STATE}" in
          PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING)
            FRAME_CACHE_JOB_ID="${CANDIDATE_JOB_ID}"
            ;;
        esac
      fi
    fi

    if [ -z "${FRAME_CACHE_JOB_ID}" ]; then
      mkdir -p "${SCRIPT_DIR}/logs_cache"
      CACHE_SBATCH_ARGS=(
        --parsable
        --partition="${FSQ_FRAME_CACHE_PARTITION}"
        --qos="${FSQ_FRAME_CACHE_QOS}"
        --cpus-per-task="${FSQ_FRAME_CACHE_CPUS_PER_TASK}"
        --mem="${FSQ_FRAME_CACHE_MEM}"
        --time="${FSQ_FRAME_CACHE_TIME}"
      )
      cd "${SCRIPT_DIR}"
      FRAME_CACHE_JOB_ID="$({ \
        PROJECT_ROOT="${PROJECT_ROOT}" \
        RAW_DATASET_DIR="${RAW_DATASET_DIR}" \
        FSQ_FRAME_CACHE_ROOT="${FSQ_FRAME_CACHE_ROOT}" \
        FSQ_FRAME_CACHE_FINGERPRINT="${FSQ_FRAME_CACHE_FINGERPRINT}" \
        FSQ_FRAME_CACHE_WORKERS="${FSQ_FRAME_CACHE_WORKERS}" \
        FSQ_FRAME_CACHE_DECODER_THREADS="${FSQ_FRAME_CACHE_DECODER_THREADS}" \
          sbatch "${CACHE_SBATCH_ARGS[@]}" "${FSQ_SRC_DIR}/prepare_fsq_frame_cache.sbatch"; \
      } | tail -1)"
      FRAME_CACHE_JOB_ID="${FRAME_CACHE_JOB_ID%%;*}"
      if ! [[ "${FRAME_CACHE_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse FSQ frame-cache Slurm job id: '${FRAME_CACHE_JOB_ID}'" >&2
        exit 1
      fi
      JOB_FILE_TMP="${FSQ_FRAME_CACHE_JOB_FILE}.tmp.$$"
      printf '%s\n' "${FRAME_CACHE_JOB_ID}" > "${JOB_FILE_TMP}"
      mv "${JOB_FILE_TMP}" "${FSQ_FRAME_CACHE_JOB_FILE}"
      echo "Submitted FSQ frame cache job ${FRAME_CACHE_JOB_ID}"
    else
      echo "Reusing active FSQ frame cache job ${FRAME_CACHE_JOB_ID}"
    fi
    FRAME_CACHE_DEPENDENCY_ARGS=(
      --dependency="afterok:${FRAME_CACHE_JOB_ID}"
      --kill-on-invalid-dep=yes
    )
  else
    echo "FSQ frame cache ready: ${FSQ_FRAME_CACHE_DIR}"
  fi
  flock -u 9
else
  FSQ_FRAME_CACHE_DIR=""
fi

cd "${SCRIPT_DIR}"
mkdir -p logs_train

echo "Submit FSQ train"
echo "  dataset     : ${TARGET_DATASET}"
echo "  FSQ inputs  : ${FSQ_INPUTS_DIR}"
echo "  skillset    : ${SKILLSET_DIR}/skills"
if [ "${FSQ_DECODER_TERMINATOR_PROGRESS}" != "true" ] && [ "${FSQ_DECODER_TERMINATOR_TERMINATION}" != "true" ]; then
  echo "  terminator  : disabled"
elif [ "${FSQ_TERMINATOR_ARCH}" = "rnn" ]; then
  echo "  terminator  : full-skill state+FSQ RNN (no image/DINO loading)"
elif [ "${FSQ_TERMINATOR_INPUT_SPACE}" = "state" ]; then
  echo "  terminator  : current state+FSQ default model (no image/DINO loading)"
else
  echo "  terminator  : ${FSQ_TERMINATOR_INPUT_SPACE}+FSQ ${FSQ_TERMINATOR_ARCH}"
  echo "  vision      : ${FSQ_VISION_BACKBONE} ← ${RAW_DATASET_DIR}/videos"
  if [ -n "${FSQ_FRAME_CACHE_DIR}" ]; then
    echo "  frame cache : ${FSQ_FRAME_CACHE_DIR}"
    echo "  local stage : ${FSQ_FRAME_CACHE_STAGE_LOCAL} (root=${FSQ_FRAME_CACHE_LOCAL_ROOT:-auto})"
  elif [ "${FSQ_FRAME_CACHE_ENABLED}" = "true" ]; then
    echo "  frame cache : waiting for job ${FRAME_CACHE_JOB_ID}"
    echo "  local stage : ${FSQ_FRAME_CACHE_STAGE_LOCAL} (root=${FSQ_FRAME_CACHE_LOCAL_ROOT:-auto})"
  else
    echo "  frame cache : disabled (live AV1 decode)"
  fi
fi
echo "  output      : ${FSQ_OUTPUT_DIR}"
echo "  slurm       : partition=${FSQ_TRAIN_PARTITION} qos=${FSQ_TRAIN_QOS} gres=${FSQ_TRAIN_GRES}"
if [ -n "${FSQ_VENV_ARCHIVE}" ]; then
  echo "  Python      : node-local copy from ${FSQ_VENV_ARCHIVE}"
else
  echo "  Python      : shared ${PROJECT_ROOT}/.venv"
fi

TRAIN_SKILLS_CONFIG="${CONFIG_PATH}" TRAIN_DATA="${TARGET_DATASET}" \
FSQ_VENV_ARCHIVE="${FSQ_VENV_ARCHIVE}" FSQ_FRAME_CACHE_DIR="${FSQ_FRAME_CACHE_DIR}" \
  sbatch "${SBATCH_ARGS[@]}" "${FRAME_CACHE_DEPENDENCY_ARGS[@]}" \
  "${FSQ_SRC_DIR}/train_fsq.sbatch"
