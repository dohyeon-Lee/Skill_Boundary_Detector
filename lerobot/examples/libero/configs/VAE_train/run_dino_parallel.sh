#!/usr/bin/env bash
# Submit parallel DINO precompute workers + auto merge.
# Nodes and GPU counts are discovered automatically from SLURM.
# ─────────────────────────────────────────────────────
# SERVER CONFIG — only edit this block when moving to a new server
PARTITIONS=("debug")    # partitions to use (nodes discovered automatically)
EXCLUDE_NODES=("node200")        # nodes to skip, e.g. ("node200" "node03")
GPU_RESERVE=1           # GPUs to leave free per node (courtesy)
GPU_MAX_PER_NODE=7      # hard cap per node
QOS=big_qos
HOMEDIR=/data2/dohyeon
PROJDIR=/SBD
VISUAL_BACKBONE=dinov3_vits16 # ${VISUAL_BACKBONE:-dinov2_small} # dinov2_small, dinov3_vits16, dinov3_convnext_small
# ─────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# Discover all nodes across given partitions.
# Fills NODE_PARTITION[node]=partition and NODE_TOTAL_GPU[node]=count.
declare -A NODE_PARTITION
declare -A NODE_TOTAL_GPU

for part in "${PARTITIONS[@]}"; do
    while read -r node gres; do
        [ -z "$node" ] && continue
        total=$(echo "$gres" | sed -nE 's/.*gpu(:[^:[:space:]]*)?:([0-9]+).*/\2/p' | head -1)
        if [ -z "$total" ] || [ "$total" -eq 0 ]; then
            echo "  skip ${node}: no parseable GPU count in GRES='${gres}'"
            continue
        fi
        [ "${NODE_PARTITION[$node]+_}" ] && continue  # already assigned to earlier partition
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

# Sort nodes for deterministic worker ID assignment
SORTED_NODES=($(echo "${!NODE_PARTITION[@]}" | tr ' ' '\n' | sort))

echo "=== GPU availability ==="
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
echo "  Total workers: ${N_WORKERS}"
echo "========================"

if [ "$N_WORKERS" -eq 0 ]; then
    echo "ERROR: No GPUs available. Try again later."
    exit 1
fi

COMMON_EXPORT="ALL,N_WORKERS=${N_WORKERS},HOMEDIR=${HOMEDIR},PROJDIR=${PROJDIR},VISUAL_BACKBONE=${VISUAL_BACKBONE}"
JOB_IDS=()
WORKER_OFFSET=0
ALL_NODES=()

for node in "${SORTED_NODES[@]}"; do
    w=${NODE_WORKERS[$node]}
    [ "$w" -eq 0 ] && continue

    part=${NODE_PARTITION[$node]}
    START=$WORKER_OFFSET
    END=$(( WORKER_OFFSET + w - 1 ))

    JOB=$(sbatch --parsable \
        --partition="$part" \
        --qos="$QOS" \
        --nodelist="$node" \
        --array="${START}-${END}" \
        --export="$COMMON_EXPORT" \
        precompute_dino_worker.sbatch)
    JOB_IDS+=("$JOB")
    ALL_NODES+=("$node")
    echo "${node} job: ${JOB}  (workers ${START}–${END})"

    WORKER_OFFSET=$(( WORKER_OFFSET + w ))
done

# Merge: pick first available node/partition for the merge job
MERGE_NODE=${ALL_NODES[0]}
MERGE_PART=${NODE_PARTITION[$MERGE_NODE]}

DEP=$(printf "afterok:%s" "${JOB_IDS[0]}")
for id in "${JOB_IDS[@]:1}"; do DEP="${DEP}:${id}"; done

MERGE_JOB=$(sbatch --parsable \
    --partition="$MERGE_PART" \
    --qos="$QOS" \
    --nodelist="$MERGE_NODE" \
    --dependency="$DEP" \
    --export="$COMMON_EXPORT" \
    precompute_dino_merge.sbatch)
echo "Merge job: ${MERGE_JOB}  (starts after all workers finish)"

echo ""
echo "Backbone: ${VISUAL_BACKBONE}"
echo "Monitor: squeue -u \$USER"
echo "Logs   : tail -f logs/DINO_worker_${JOB_IDS[0]}_0.err"
