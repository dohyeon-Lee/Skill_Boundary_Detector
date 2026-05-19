#!/usr/bin/env bash
# Submit parallel frame-level DINO/DINOv3 precompute workers for raw datasets.
# Workers shard episodes by worker_id modulo N_WORKERS and write per-episode
# npz files, so there is no merge job.

# ── Server / job defaults ────────────────────────────────────────────────────
PARTITIONS=("debug")
EXCLUDE_NODES=("node200")
# PARTITIONS=("base_suma_rtx3090" "dell_rtx3090" "big_suma_rtx3090") #"suma_a6000" "asus_6000ada" "suma_rtx4090" )
# EXCLUDE_NODES=(node19 node13 node18 node16 node08 node10 node21 node14 node04 node05 node31)
GPU_RESERVE=0
GPU_MAX_PER_NODE=7
QOS=big_qos

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

# HOMEDIR=/scratch/mdorazi
# PROJDIR=/Skill_Boundary_Detector

# ── Precompute defaults ──────────────────────────────────────────────────────
DATASET=libero_90
DATASET_ROOT=libero_dataset
VISUAL_BACKBONE=dinov3_vits16  # dinov3_vits16, dinov2_small
IMAGE_KEYS=observation.images.image
PATCH_GRID=8
IMAGE_SIZE=224
BATCH_SIZE=1024
DTYPE=float16
WANDB_PROJECT=DP_train

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

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

echo "=== Frame DINO GPU plan ==="
declare -A NODE_WORKERS
declare -A NODE_FREE_GPU
N_WORKERS=0
for node in "${SORTED_NODES[@]}"; do
    # Only target idle/mixed nodes; skip alloc/down/drain
    state=$(sinfo -n "$node" -h -o "%T" 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]')
    if [[ "$state" != "idle" && "$state" != "mixed" ]]; then
        echo "  ${node}: state=${state:-unknown}, skipping"
        continue
    fi

    total=${NODE_TOTAL_GPU[$node]}
    alloc=$(scontrol show node "$node" 2>/dev/null \
        | grep -oP 'AllocTRES=\S+' \
        | grep -oP 'gres/gpu=\K[0-9]+' || true)
    [ -z "$alloc" ] && alloc=0
    free=$(( total - alloc ))

    w=$(( total - GPU_RESERVE ))
    [ "$w" -lt 0 ] && w=0
    [ "$w" -gt "$GPU_MAX_PER_NODE" ] && w=$GPU_MAX_PER_NODE
    NODE_WORKERS[$node]=$w
    NODE_FREE_GPU[$node]=$free
    echo "  ${node} (${NODE_PARTITION[$node]}): ${free}/${total} free  ->  ${w} workers  [state: ${state}]"
    N_WORKERS=$(( N_WORKERS + w ))
done
echo "  Total workers: ${N_WORKERS}"
echo "==================================="

# Re-sort nodes by free GPUs descending so workers are assigned to most-available nodes first
mapfile -t SORTED_NODES < <(
    for node in "${!NODE_FREE_GPU[@]}"; do
        echo "${NODE_FREE_GPU[$node]} $node"
    done | sort -rn | awk '{print $2}'
)

if [ "$N_WORKERS" -eq 0 ]; then
    echo "ERROR: No GPUs available. Try again later."
    exit 1
fi

# Cap N_WORKERS to actual number of video files (sharding is per video file, not per episode)
VIDEO_FILE_COUNT=$(find "${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET}/videos" -name "*.mp4" 2>/dev/null | wc -l)
if [ "$VIDEO_FILE_COUNT" -gt 0 ] && [ "$N_WORKERS" -gt "$VIDEO_FILE_COUNT" ]; then
    echo "  Capping N_WORKERS: ${N_WORKERS} -> ${VIDEO_FILE_COUNT} (only ${VIDEO_FILE_COUNT} video files in dataset)"
    N_WORKERS=$VIDEO_FILE_COUNT
    remaining=$N_WORKERS
    for node in "${SORTED_NODES[@]}"; do
        [[ -v NODE_WORKERS[$node] ]] || continue
        old_w=${NODE_WORKERS[$node]:-0}
        new_w=$(( old_w < remaining ? old_w : remaining ))
        NODE_WORKERS[$node]=$new_w
        remaining=$(( remaining - new_w ))
    done
fi

COMMON_EXPORT=ALL
COMMON_EXPORT+=",N_WORKERS=${N_WORKERS}"
COMMON_EXPORT+=",HOMEDIR=${HOMEDIR}"
COMMON_EXPORT+=",PROJDIR=${PROJDIR}"
COMMON_EXPORT+=",DATASET=${DATASET}"
COMMON_EXPORT+=",DATASET_ROOT=${DATASET_ROOT}"
COMMON_EXPORT+=",VISUAL_BACKBONE=${VISUAL_BACKBONE}"
COMMON_EXPORT+=",IMAGE_KEYS=${IMAGE_KEYS//,/:}"
COMMON_EXPORT+=",PATCH_GRID=${PATCH_GRID}"
COMMON_EXPORT+=",IMAGE_SIZE=${IMAGE_SIZE}"
COMMON_EXPORT+=",BATCH_SIZE=${BATCH_SIZE}"
COMMON_EXPORT+=",DTYPE=${DTYPE}"
COMMON_EXPORT+=",WANDB_PROJECT=${WANDB_PROJECT}"

JOB_IDS=()
WORKER_OFFSET=0

for part in "${PARTITIONS[@]}"; do
    part_workers=0
    for node in "${SORTED_NODES[@]}"; do
        w=${NODE_WORKERS[$node]:-0}
        [ "$w" -eq 0 ] && continue
        [ "${NODE_PARTITION[$node]:-}" = "$part" ] && part_workers=$(( part_workers + w ))
    done
    [ "$part_workers" -eq 0 ] && continue

    # Build exclude list with only nodes that belong to this partition
    part_nodes=$(sinfo -p "$part" -N -h -o "%N" 2>/dev/null | sort -u)
    part_exclude=""
    for ex in "${EXCLUDE_NODES[@]}"; do
        echo "$part_nodes" | grep -qx "$ex" && part_exclude="${part_exclude:+$part_exclude,}$ex"
    done

    start=$WORKER_OFFSET
    end=$(( WORKER_OFFSET + part_workers - 1 ))

    job=$(sbatch --parsable \
        --partition="$part" \
        --qos="$QOS" \
        ${part_exclude:+--exclude="$part_exclude"} \
        --array="${start}-${end}" \
        --export="$COMMON_EXPORT" \
        precompute_frame_dino.sbatch)
    JOB_IDS+=("$job")
    echo "${part} job: ${job}  (workers ${start}-${end})"

    WORKER_OFFSET=$(( WORKER_OFFSET + part_workers ))
done

echo ""
echo "Submitted ${#JOB_IDS[@]} array job(s), total workers=${N_WORKERS}"
echo "Dataset : ${DATASET_ROOT}/${DATASET}"
echo "Backbone: ${VISUAL_BACKBONE}"
echo "Cameras : ${IMAGE_KEYS}"
echo "Output  : ${HOMEDIR}${PROJDIR}/${DATASET_ROOT}/${DATASET}_data/${DATASET}_DINO"
echo "Monitor : squeue -u ${USER}"
echo "Logs    : tail -f logs/DP_frame_DINO_${JOB_IDS[0]}_0.err"
