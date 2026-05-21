#!/usr/bin/env bash
# Stage 3: build FSQ inputs from existing frame DINO features and skillset data.
#
# This script does not run DINO. It expects Stage 1_DINO to have produced
# frame-level CLS/pooled-patch features, and Stage 2_skillset to have produced
# the per-skill dataset. It then:
#   1. extracts skill-level DINO tokens into one npz,
#   2. runs SAM2 workers for per-skill patch flags,
#   3. merges those flags into one npz.

set -euo pipefail

# Server config
PARTITIONS=("debug")
EXCLUDE_NODES=("node200")

# PARTITIONS=(base_suma_rtx3090 dell_rtx3090 big_suma_rtx3090 suma_a6000 suma_rtx4090)
# EXCLUDE_NODES=(node19 node13 node18 node16 node08 node10 node21 node14 node04 node05 node31 node28)

GPU_RESERVE=0
GPU_MAX_PER_NODE=7
QOS=big_qos

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

CONFIG_PY="${SCRIPT_DIR}/../pipeline_config.py"
eval "$(python3 "${CONFIG_PY}" --shell)"

DERIVED_DATA_DIR=${DATA_DIR}
FRAME_DINO_DIR=${FRAME_DINO_DIR}
SAM2_OUTPUT_DIR=${SAM2_MASKS_DIR}
SAM2_MERGED_PATH=${SAM2_FLAGS_PATH}

mkdir -p "${FSQ_PRECOMPUTE_DIR}"

if [ ! -d "${SKILLSET_DIR}/skills" ]; then
  echo "ERROR: skillset directory not found:"
  echo "  ${SKILLSET_DIR}/skills"
  echo "Run data_generation/2_skillset/build_skill_dataset.sbatch first."
  exit 1
fi

if [ ! -d "${FRAME_DINO_DIR}" ]; then
  echo "ERROR: frame DINO directory not found:"
  echo "  ${FRAME_DINO_DIR}"
  echo "Run data_generation/1_DINO/run_frame_dino_parallel.sh first."
  exit 1
fi

if [ ! -f "${SAM2_CHECKPOINT}" ]; then
  echo "ERROR: SAM2 checkpoint not found:"
  echo "  ${SAM2_CHECKPOINT}"
  exit 1
fi

echo "FSQ precompute"
echo "  dataset       : ${DATA}"
echo "  skillset      : ${SKILLSET_DIR}/skills"
echo "  frame DINO    : ${FRAME_DINO_DIR}"
echo "  DINO tokens   : ${DINO_TOKENS_PATH}"
echo "  SAM2 masks    : ${SAM2_OUTPUT_DIR}"
echo "  SAM2 flags    : ${SAM2_MERGED_PATH}"
echo ""

declare -A NODE_PARTITION
declare -A NODE_TOTAL_GPU

for part in "${PARTITIONS[@]}"; do
  while read -r node gres; do
    [ -z "${node}" ] && continue
    total=$(echo "${gres}" | sed -nE 's/.*gpu(:[^:[:space:]]*)?:([0-9]+).*/\2/p' | head -1)
    if [ -z "${total}" ] || [ "${total}" -eq 0 ]; then continue; fi
    [ "${NODE_PARTITION[$node]+_}" ] && continue
    for ex in "${EXCLUDE_NODES[@]+"${EXCLUDE_NODES[@]}"}"; do
      [ "${node}" = "${ex}" ] && continue 2
    done
    NODE_PARTITION[$node]=${part}
    NODE_TOTAL_GPU[$node]=${total}
  done < <(sinfo -p "${part}" -N -o "%N %G" --noheader 2>/dev/null)
done

# N_WORKERS = total GPUs across all nodes (capped per node), regardless of current availability.
# SLURM will queue the jobs and run them when GPUs become free.
N_WORKERS=0
for node in "${!NODE_TOTAL_GPU[@]}"; do
  total=${NODE_TOTAL_GPU[$node]}
  w=$(( total - GPU_RESERVE ))
  [ $w -lt 0 ] && w=0
  [ $w -gt $GPU_MAX_PER_NODE ] && w=$GPU_MAX_PER_NODE
  N_WORKERS=$(( N_WORKERS + w ))
done

if [ "${N_WORKERS}" -eq 0 ]; then
  echo "ERROR: No GPU nodes found in partitions: ${PARTITIONS[*]}"
  exit 1
fi

echo "SAM2 workers to submit: ${N_WORKERS}  (queued until GPUs free)"
echo ""

COMMON_EXPORT="ALL,N_WORKERS=${N_WORKERS},HOMEDIR=${HOMEDIR},PROJDIR=${PROJDIR}"
COMMON_EXPORT+=",VISUAL_BACKBONE=${VISUAL_BACKBONE},PATCH_GRID=${PATCH_GRID}"
COMMON_EXPORT+=",DATA=${DATA},DATADIR=${DATADIR},SKILLSET=${SKILLSET},DERIVED_DATA_DIR=${DERIVED_DATA_DIR}"
COMMON_EXPORT+=",FSQ_PRECOMPUTE_DIR=${FSQ_PRECOMPUTE_DIR},SKILLSET_DIR=${SKILLSET_DIR}"
COMMON_EXPORT+=",DINO_TOKENS_PATH=${DINO_TOKENS_PATH},SAM2_OUTPUT_DIR=${SAM2_OUTPUT_DIR}"
COMMON_EXPORT+=",SAM2_MERGED_PATH=${SAM2_MERGED_PATH},SAM2_CHECKPOINT=${SAM2_CHECKPOINT}"

FIRST_PART="${PARTITIONS[0]}"

EXTRACT_JOB=$(sbatch --parsable \
  --partition="${FIRST_PART}" \
  --qos="${QOS}" \
  --export="${COMMON_EXPORT}" \
  extract_skill_tokens.sbatch)
echo "Skill token extraction job: ${EXTRACT_JOB}"

SAM2_JOB=$(sbatch --parsable \
  --partition="${FIRST_PART}" \
  --qos="${QOS}" \
  --array="0-$(( N_WORKERS - 1 ))" \
  --export="${COMMON_EXPORT}" \
  precompute_sam2_masks_worker.sbatch)
echo "SAM2 mask workers job:      ${SAM2_JOB}  (array 0–$(( N_WORKERS - 1 )), queued)"

MERGE_JOB=$(sbatch --parsable \
  --partition="${FIRST_PART}" \
  --qos="${QOS}" \
  --dependency="afterok:${SAM2_JOB}" \
  --export="${COMMON_EXPORT}" \
  merge_sam2_patch_flags.sbatch)
echo "SAM2 merge job:             ${MERGE_JOB}  (runs after all SAM2 workers finish)"

echo ""
echo "Backbone : ${VISUAL_BACKBONE}  patch_grid=${PATCH_GRID}"
echo "Monitor  : squeue -u \$USER"
echo "Logs     : ${SCRIPT_DIR}/logs"
