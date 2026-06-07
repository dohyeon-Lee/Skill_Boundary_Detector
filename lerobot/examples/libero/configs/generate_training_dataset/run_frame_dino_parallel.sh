#!/usr/bin/env bash
# Inputs:
#   dataset name : --dataset or dino_target_dataset in training_dataset_config.yaml
#   dataset path : {project_root}/{dataset_root}/{dataset}
# Reference model:
#   DINO model   : {project_root}/models/dinov3-vits16
# Outputs:
#   DINO name    : {dataset}_DINO/pg{dino_patch_grid}
#   DINO path    : {project_root}/{dataset_root}/{dataset}_DINO/pg{dino_patch_grid}
#
# Submit frame-level DINO/DINOv3 precompute jobs for a base training dataset.
#
# Defaults come from training_dataset_config.yaml. Command-line flags override
# the yaml for this run:
#   ./run_frame_dino_parallel.sh --dataset libero_90                 # 3rd-person (yaml default)
#   ./run_frame_dino_parallel.sh --dataset libero_90 --camera wrist  # wrist only (separate run)
#   ./run_frame_dino_parallel.sh --dataset libero_90 --camera both   # both cameras in one run
#   ./run_frame_dino_parallel.sh --dataset libero_10 --config ./training_dataset_config.yaml
#
# --camera picks which camera(s) this run precomputes (independent runs write to the same
#   {dataset}_DINO/pg{patch_grid}/ dir, one subdir per camera):
#     image  -> observation.images.image        (3rd-person)
#     wrist  -> observation.images.wrist_image
#     both   -> both
#     (anything else is treated as a raw comma-separated image-key list)
#   Omit --camera to use the yaml's dino_image_keys (the DP path stays 3rd-person).
#
# Output:
#   {project_root}/{dataset_root}/{dataset}_DINO/pg{patch_grid}/<camera_subdir>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${SCRIPT_DIR}/training_dataset_config.yaml"
TARGET_DATASET=""
CAMERA=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dataset)
      TARGET_DATASET="$2"
      shift 2
      ;;
    --camera)
      CAMERA="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,33p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BOOTSTRAP_PYTHON="${PROJECT_ROOT:-}/.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../.venv/bin/python"
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TARGET_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/training_dataset_config.py" --config "${CONFIG_PATH}" --shell-dino --dataset "${TARGET_DATASET}")"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/training_dataset_config.py" --config "${CONFIG_PATH}" --shell-dino)"
fi

# --camera overrides which camera(s) this run precomputes (yaml dino_image_keys otherwise).
case "${CAMERA}" in
  "")                       ;;  # keep yaml IMAGE_KEYS
  image|third|3rd)         IMAGE_KEYS="observation.images.image" ;;
  wrist|eye_in_hand)       IMAGE_KEYS="observation.images.wrist_image" ;;
  both|all)                IMAGE_KEYS="observation.images.image,observation.images.wrist_image" ;;
  *)                       IMAGE_KEYS="${CAMERA}" ;;  # raw comma-separated image-key list
esac

cd "${SCRIPT_DIR}"
mkdir -p logs

if [ ! -d "${DATASET_DIR}" ]; then
  echo "Dataset not found: ${DATASET_DIR}" >&2
  exit 1
fi

if [ ! -d "${IMAGE_MODEL_PATH}" ]; then
  echo "Image model not found: ${IMAGE_MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a PARTITIONS <<< "${PARTITIONS}"
IFS=',' read -r -a EXCLUDE_NODES <<< "${EXCLUDE_NODES:-}"

declare -A NODE_PARTITION
declare -A NODE_TOTAL_GPU

for part in "${PARTITIONS[@]}"; do
  [ -z "${part}" ] && continue
  while read -r node gres; do
    [ -z "${node}" ] && continue
    total=$(echo "${gres}" | sed -nE 's/.*gpu(:[^:[:space:]]*)?:([0-9]+).*/\2/p' | head -1)
    if [ -z "${total}" ] || [ "${total}" -eq 0 ]; then
      continue
    fi
    [ "${NODE_PARTITION[$node]+_}" ] && continue
    for ex in "${EXCLUDE_NODES[@]}"; do
      [ -n "${ex}" ] && [ "${node}" = "${ex}" ] && continue 2
    done
    NODE_PARTITION[$node]=${part}
    NODE_TOTAL_GPU[$node]=${total}
  done < <(sinfo -p "${part}" -N -o "%N %G" --noheader 2>/dev/null)
done

if [ "${#NODE_PARTITION[@]}" -eq 0 ]; then
  echo "ERROR: No GPU nodes found in partitions: ${PARTITIONS[*]}"
  exit 1
fi

mapfile -t SORTED_NODES < <(printf "%s\n" "${!NODE_PARTITION[@]}" | sort)

echo "=== Frame DINO GPU plan ==="
declare -A NODE_WORKERS
declare -A NODE_FREE_GPU
N_WORKERS=0
for node in "${SORTED_NODES[@]}"; do
  state=$(sinfo -n "${node}" -h -o "%T" 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]')
  if [[ "${state}" != "idle" && "${state}" != "mixed" ]]; then
    echo "  ${node}: state=${state:-unknown}, skipping"
    continue
  fi

  total=${NODE_TOTAL_GPU[$node]}
  alloc=$(scontrol show node "${node}" 2>/dev/null \
    | grep -oP 'AllocTRES=\S+' \
    | grep -oP 'gres/gpu=\K[0-9]+' || true)
  [ -z "${alloc}" ] && alloc=0
  free=$(( total - alloc ))

  w=$(( total - GPU_RESERVE ))
  [ "${w}" -lt 0 ] && w=0
  [ "${w}" -gt "${GPU_MAX_PER_NODE}" ] && w=${GPU_MAX_PER_NODE}
  NODE_WORKERS[$node]=${w}
  NODE_FREE_GPU[$node]=${free}
  echo "  ${node} (${NODE_PARTITION[$node]}): ${free}/${total} free -> ${w} workers [state: ${state}]"
  N_WORKERS=$(( N_WORKERS + w ))
done

if [ "${N_WORKERS}" -eq 0 ]; then
  echo "ERROR: No GPUs available. Try again later."
  exit 1
fi

if [ "${MAX_WORKERS}" -gt 0 ] && [ "${N_WORKERS}" -gt "${MAX_WORKERS}" ]; then
  echo "  Capping N_WORKERS: ${N_WORKERS} -> ${MAX_WORKERS} (dino_max_workers)"
  N_WORKERS=${MAX_WORKERS}
fi

echo "  Counting dataset shards..."
VIDEO_FILE_COUNT=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
if [ "${VIDEO_FILE_COUNT}" -gt 0 ] && [ "${N_WORKERS}" -gt "${VIDEO_FILE_COUNT}" ]; then
  echo "  Capping N_WORKERS: ${N_WORKERS} -> ${VIDEO_FILE_COUNT} (video files)"
  N_WORKERS=${VIDEO_FILE_COUNT}
fi
if [ "${VIDEO_FILE_COUNT}" -eq 0 ]; then
  EPISODE_COUNT="$("${BOOTSTRAP_PYTHON}" - "${DATASET_DIR}" <<'PY'
import json, sys
from pathlib import Path
info = Path(sys.argv[1]) / "meta" / "info.json"
print(json.loads(info.read_text()).get("total_episodes", 0) if info.exists() else 0)
PY
)"
  echo "  Detected image/parquet dataset (no mp4 videos); episodes=${EPISODE_COUNT}"
  if [ "${EPISODE_COUNT}" -gt 0 ] && [ "${N_WORKERS}" -gt "${EPISODE_COUNT}" ]; then
    echo "  Capping N_WORKERS: ${N_WORKERS} -> ${EPISODE_COUNT} (episodes)"
    N_WORKERS=${EPISODE_COUNT}
  fi
fi

mapfile -t SORTED_NODES < <(
  for node in "${!NODE_FREE_GPU[@]}"; do
    echo "${NODE_FREE_GPU[$node]} ${node}"
  done | sort -rn | awk '{print $2}'
)

remaining=${N_WORKERS}
for node in "${SORTED_NODES[@]}"; do
  old_w=${NODE_WORKERS[$node]:-0}
  new_w=$(( old_w < remaining ? old_w : remaining ))
  NODE_WORKERS[$node]=${new_w}
  remaining=$(( remaining - new_w ))
done

echo "  Total workers: ${N_WORKERS}"
echo "==================================="

IMAGE_KEYS_EXPORT="${IMAGE_KEYS//,/:}"
COMMON_EXPORT="ALL"
COMMON_EXPORT+=",N_WORKERS=${N_WORKERS}"
COMMON_EXPORT+=",PROJECT_ROOT=${PROJECT_ROOT}"
COMMON_EXPORT+=",DATASET=${DATASET}"
COMMON_EXPORT+=",DATASET_DIR=${DATASET_DIR}"
COMMON_EXPORT+=",OUTPUT_DIR=${OUTPUT_DIR}"
COMMON_EXPORT+=",VISUAL_BACKBONE=${VISUAL_BACKBONE}"
COMMON_EXPORT+=",IMAGE_MODEL_PATH=${IMAGE_MODEL_PATH}"
COMMON_EXPORT+=",N_PATCH_RAW=${N_PATCH_RAW}"
COMMON_EXPORT+=",IMAGE_KEYS=${IMAGE_KEYS_EXPORT}"
COMMON_EXPORT+=",PATCH_GRID=${PATCH_GRID}"
COMMON_EXPORT+=",IMAGE_SIZE=${IMAGE_SIZE}"
COMMON_EXPORT+=",BATCH_SIZE=${BATCH_SIZE}"
COMMON_EXPORT+=",DTYPE=${DTYPE}"
COMMON_EXPORT+=",WANDB_PROJECT=${WANDB_PROJECT}"

JOB_IDS=()
WORKER_OFFSET=0

for part in "${PARTITIONS[@]}"; do
  [ -z "${part}" ] && continue
  part_workers=0
  for node in "${SORTED_NODES[@]}"; do
    w=${NODE_WORKERS[$node]:-0}
    [ "${w}" -eq 0 ] && continue
    [ "${NODE_PARTITION[$node]:-}" = "${part}" ] && part_workers=$(( part_workers + w ))
  done
  [ "${part_workers}" -eq 0 ] && continue

  part_nodes=$(sinfo -p "${part}" -N -h -o "%N" 2>/dev/null | sort -u)
  part_exclude=""
  for ex in "${EXCLUDE_NODES[@]}"; do
    [ -z "${ex}" ] && continue
    echo "${part_nodes}" | grep -qx "${ex}" && part_exclude="${part_exclude:+${part_exclude},}${ex}"
  done

  start=${WORKER_OFFSET}
  end=$(( WORKER_OFFSET + part_workers - 1 ))

  job=$(sbatch --parsable \
    --partition="${part}" \
    --qos="${QOS}" \
    ${part_exclude:+--exclude="${part_exclude}"} \
    --array="${start}-${end}" \
    --export="${COMMON_EXPORT}" \
    "${SRC_DIR}/precompute_frame_dino_worker.sbatch")
  JOB_IDS+=("${job}")
  echo "${part} job: ${job} (workers ${start}-${end})"

  WORKER_OFFSET=$(( WORKER_OFFSET + part_workers ))
done

echo ""
echo "Submitted ${#JOB_IDS[@]} array job(s), total workers=${N_WORKERS}"
echo "Dataset : ${DATASET_DIR}"
echo "Backbone: ${VISUAL_BACKBONE}"
echo "Cameras : ${IMAGE_KEYS}"
echo "Output  : ${OUTPUT_DIR}"
echo "Monitor : squeue -u ${USER}"
