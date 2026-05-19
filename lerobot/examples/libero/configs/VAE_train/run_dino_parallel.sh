#!/usr/bin/env bash
# Submit skill DINO token extraction + SAM2 mask workers.
#
# DINO tokens are extracted from frame-level features already computed by
# DP_train/run_frame_dino_parallel.sh — no GPU re-encoding needed.
# SAM2 masks are computed directly (GPU, parallel workers).
#
# Prerequisites:
#   DP_train/run_frame_dino_parallel.sh must have completed for DATA/VISUAL_BACKBONE.
# ─────────────────────────────────────────────────────────────────────────────

# ── Server config ─────────────────────────────────────────────────────────────
PARTITIONS=("debug")
EXCLUDE_NODES=("node200")
GPU_RESERVE=1
GPU_MAX_PER_NODE=7
QOS=big_qos

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

# ── Precompute settings ───────────────────────────────────────────────────────
VISUAL_BACKBONE=dinov3_vits16   # must match what run_frame_dino_parallel.sh used
PATCH_GRID=8
DATA=libero_90
DATADIR=libero_dataset
SKILLSET=${DATA}_skillset
DERIVED_DATA_DIR=${HOMEDIR}${PROJDIR}/${DATADIR}/${DATA}_data
FSQ_PRECOMPUTE_DIR=${DERIVED_DATA_DIR}/${DATA}_for_FSQ
SKILLSET_DIR=${FSQ_PRECOMPUTE_DIR}/${SKILLSET}
SAM2_OUTPUT_DIR=${FSQ_PRECOMPUTE_DIR}/sam2_masks
SAM2_MERGED_PATH=${FSQ_PRECOMPUTE_DIR}/patch_flags.npz
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Resolve backbone paths (sets IMAGE_FEATURE_TAG, DINO_TOKENS_PATH, etc.)
source visual_backbone.sh
resolve_visual_backbone

FRAME_DINO_DIR="${DERIVED_DATA_DIR}/${DATA}_DINO/${IMAGE_FEATURE_TAG}_pg${PATCH_GRID}"
DINO_TOKENS_PATH=${FSQ_PRECOMPUTE_DIR}/${IMAGE_FEATURE_TAG}_tokens.npz
mkdir -p "${FSQ_PRECOMPUTE_DIR}"

# Check frame DINO exists before submitting anything
if [ ! -d "${FRAME_DINO_DIR}" ]; then
  echo "ERROR: Frame DINO directory not found:"
  echo "  ${FRAME_DINO_DIR}"
  echo "Run DP_train/run_frame_dino_parallel.sh for dataset '${DATA}' first."
  exit 1
fi
echo "Frame DINO dir : ${FRAME_DINO_DIR}  [OK]"
echo "Skillset dir   : ${SKILLSET_DIR}"
echo "Skill tokens   : ${DINO_TOKENS_PATH}"
echo "SAM2 masks     : ${SAM2_OUTPUT_DIR}"
echo "SAM2 flags     : ${SAM2_MERGED_PATH}"
echo ""

# ── Count total GPUs per partition (for N_WORKERS, no availability check) ────
declare -A NODE_PARTITION
declare -A NODE_TOTAL_GPU

for part in "${PARTITIONS[@]}"; do
    while read -r node gres; do
        [ -z "$node" ] && continue
        total=$(echo "$gres" | sed -nE 's/.*gpu(:[^:[:space:]]*)?:([0-9]+).*/\2/p' | head -1)
        if [ -z "$total" ] || [ "$total" -eq 0 ]; then continue; fi
        [ "${NODE_PARTITION[$node]+_}" ] && continue
        for ex in "${EXCLUDE_NODES[@]+"${EXCLUDE_NODES[@]}"}"; do
            [ "$node" = "$ex" ] && continue 2
        done
        NODE_PARTITION[$node]=$part
        NODE_TOTAL_GPU[$node]=$total
    done < <(sinfo -p "$part" -N -o "%N %G" --noheader 2>/dev/null)
done

if [ ${#NODE_PARTITION[@]} -eq 0 ]; then
    echo "ERROR: No GPU nodes found in partitions: ${PARTITIONS[*]}"
    exit 1
fi

# N_WORKERS = total GPUs across all nodes (capped per node), regardless of current availability.
# SLURM will queue the jobs and run them when GPUs become free.
N_WORKERS=0
for node in "${!NODE_PARTITION[@]}"; do
    total=${NODE_TOTAL_GPU[$node]}
    w=$(( total - GPU_RESERVE ))
    [ $w -lt 0 ] && w=0
    [ $w -gt $GPU_MAX_PER_NODE ] && w=$GPU_MAX_PER_NODE
    N_WORKERS=$(( N_WORKERS + w ))
done

echo "SAM2 workers to submit: ${N_WORKERS}  (queued until GPUs free)"
echo ""

COMMON_EXPORT="ALL,N_WORKERS=${N_WORKERS},HOMEDIR=${HOMEDIR},PROJDIR=${PROJDIR}"
COMMON_EXPORT+=",VISUAL_BACKBONE=${VISUAL_BACKBONE},PATCH_GRID=${PATCH_GRID}"
COMMON_EXPORT+=",DATA=${DATA},DATADIR=${DATADIR},SAM2_OUTPUT_DIR=${SAM2_OUTPUT_DIR}"
COMMON_EXPORT+=",DERIVED_DATA_DIR=${DERIVED_DATA_DIR},FSQ_PRECOMPUTE_DIR=${FSQ_PRECOMPUTE_DIR}"
COMMON_EXPORT+=",SKILLSET_DIR=${SKILLSET_DIR},DINO_TOKENS_PATH=${DINO_TOKENS_PATH}"
COMMON_EXPORT+=",SAM2_MERGED_PATH=${SAM2_MERGED_PATH}"

# Pick any node in the first partition for CPU-only jobs
FIRST_PART="${PARTITIONS[0]}"
FIRST_NODE=$(printf "%s\n" "${!NODE_PARTITION[@]}" | sort | head -1)

# ── Skill DINO token extraction (CPU, single job) ────────────────────────────
EXTRACT_JOB=$(sbatch --parsable \
    --partition="$FIRST_PART" \
    --qos="$QOS" \
    --export="$COMMON_EXPORT" \
    extract_skill_tokens.sbatch)
echo "Skill token extraction job: ${EXTRACT_JOB}  (CPU-only, any node)"

# ── SAM2 mask workers (GPU, queued — runs when GPUs become available) ─────────
if [ "$N_WORKERS" -eq 0 ]; then
    echo "ERROR: No GPU nodes configured in partitions: ${PARTITIONS[*]}"
    exit 1
fi

SAM2_JOB=$(sbatch --parsable \
    --partition="$FIRST_PART" \
    --qos="$QOS" \
    --array="0-$(( N_WORKERS - 1 ))" \
    --export="$COMMON_EXPORT" \
    precompute_sam2_masks_worker.sbatch)
echo "SAM2 mask workers job:      ${SAM2_JOB}  (array 0–$(( N_WORKERS - 1 )), queued)"

MERGE_JOB=$(sbatch --parsable \
    --partition="$FIRST_PART" \
    --qos="$QOS" \
    --dependency="afterok:${SAM2_JOB}" \
    --export="$COMMON_EXPORT" \
    merge_sam2_patch_flags.sbatch)
echo "SAM2 merge job:             ${MERGE_JOB}  (runs after all SAM2 workers finish)"

echo ""
echo "Backbone : ${VISUAL_BACKBONE}  patch_grid=${PATCH_GRID}"
echo "Output   : ${SAM2_OUTPUT_DIR}"
echo "Flags    : ${SAM2_MERGED_PATH}"
echo "Monitor  : squeue -u \$USER"
echo "Logs     : tail -f logs/skill_tokens_${EXTRACT_JOB}.out"
