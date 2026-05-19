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
SAM2_OUTPUT_DIR=${HOMEDIR}${PROJDIR}/outputs/${SKILLSET}_FSQ
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Resolve backbone paths (sets IMAGE_FEATURE_TAG, DINO_TOKENS_PATH, etc.)
source visual_backbone.sh
resolve_visual_backbone

FRAME_DINO_DIR="${HOMEDIR}${PROJDIR}/outputs/${DATA}_frame_dino_features/${IMAGE_FEATURE_TAG}_pg${PATCH_GRID}"

# Check frame DINO exists before submitting anything
if [ ! -d "${FRAME_DINO_DIR}" ]; then
  echo "ERROR: Frame DINO directory not found:"
  echo "  ${FRAME_DINO_DIR}"
  echo "Run DP_train/run_frame_dino_parallel.sh for dataset '${DATA}' first."
  exit 1
fi
echo "Frame DINO dir : ${FRAME_DINO_DIR}  [OK]"
echo "Skill tokens   : ${DINO_TOKENS_PATH}"
echo ""

# ── Discover GPU nodes for SAM2 ───────────────────────────────────────────────
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

mapfile -t SORTED_NODES < <(printf "%s\n" "${!NODE_PARTITION[@]}" | sort)

echo "=== GPU availability (for SAM2) ==="
declare -A NODE_WORKERS
N_WORKERS=0
for node in "${SORTED_NODES[@]}"; do
    total=${NODE_TOTAL_GPU[$node]}
    used=$(squeue -w "$node" -h -o "%b" 2>/dev/null | grep -c "gpu" || true)
    free=$(( total - used ))
    w=$(( free - GPU_RESERVE ))
    [ $w -lt 0 ] && w=0
    [ $w -gt $GPU_MAX_PER_NODE ] && w=$GPU_MAX_PER_NODE
    NODE_WORKERS[$node]=$w
    echo "  ${node} (${NODE_PARTITION[$node]}): ${free}/${total} free  →  ${w} workers"
    N_WORKERS=$(( N_WORKERS + w ))
done
echo "  Total SAM2 workers: ${N_WORKERS}"
echo "==================================="
echo ""

COMMON_EXPORT="ALL,N_WORKERS=${N_WORKERS},HOMEDIR=${HOMEDIR},PROJDIR=${PROJDIR}"
COMMON_EXPORT+=",VISUAL_BACKBONE=${VISUAL_BACKBONE},PATCH_GRID=${PATCH_GRID}"
COMMON_EXPORT+=",DATA=${DATA},DATADIR=${DATADIR},SAM2_OUTPUT_DIR=${SAM2_OUTPUT_DIR}"

# ── Skill DINO token extraction (CPU, single job) ────────────────────────────
EXTRACT_NODE=${SORTED_NODES[0]}
EXTRACT_PART=${NODE_PARTITION[$EXTRACT_NODE]}

EXTRACT_JOB=$(sbatch --parsable \
    --partition="$EXTRACT_PART" \
    --qos="$QOS" \
    --nodelist="$EXTRACT_NODE" \
    --export="$COMMON_EXPORT" \
    extract_skill_tokens.sbatch)
echo "Skill token extraction job: ${EXTRACT_JOB}  (${EXTRACT_NODE}, CPU-only)"

# ── SAM2 mask workers (GPU, parallel) ────────────────────────────────────────
if [ "$N_WORKERS" -eq 0 ]; then
    echo "WARNING: No GPUs available — SAM2 workers not submitted."
else
    echo ""
    echo "=== Submitting SAM2 mask workers ==="
    SAM2_JOB_IDS=()
    SAM2_WORKER_OFFSET=0

    for node in "${SORTED_NODES[@]}"; do
        w=${NODE_WORKERS[$node]}
        [ "$w" -eq 0 ] && continue

        part=${NODE_PARTITION[$node]}
        START=$SAM2_WORKER_OFFSET
        END=$(( SAM2_WORKER_OFFSET + w - 1 ))

        JOB=$(sbatch --parsable \
            --partition="$part" \
            --qos="$QOS" \
            --nodelist="$node" \
            --array="${START}-${END}" \
            --export="$COMMON_EXPORT" \
            precompute_sam2_masks_worker.sbatch)
        SAM2_JOB_IDS+=("$JOB")
        echo "  ${node} SAM2 job: ${JOB}  (workers ${START}–${END})"

        SAM2_WORKER_OFFSET=$(( SAM2_WORKER_OFFSET + w ))
    done
    echo "  SAM2 output: ${SAM2_OUTPUT_DIR}"
fi

echo ""
echo "Backbone : ${VISUAL_BACKBONE}  patch_grid=${PATCH_GRID}"
echo "Monitor  : squeue -u \$USER"
echo "Logs     : tail -f logs/skill_tokens_${EXTRACT_JOB}.out"
