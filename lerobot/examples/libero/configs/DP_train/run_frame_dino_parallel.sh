#!/usr/bin/env bash
# Submit parallel frame-level DINO/DINOv3 precompute workers for raw datasets.
# Workers shard episodes by worker_id modulo N_WORKERS and write per-episode
# npz files, so there is no merge job.

# ── Server / job defaults ────────────────────────────────────────────────────
PARTITIONS=("debug")
EXCLUDE_NODES=("node200")
GPU_RESERVE=1
GPU_MAX_PER_NODE=7
QOS=big_qos

HOMEDIR=/data2/dohyeon
PROJDIR=/SBD

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

echo "=== Frame DINO GPU availability ==="
declare -A NODE_WORKERS
N_WORKERS=0
for node in "${SORTED_NODES[@]}"; do
    total=${NODE_TOTAL_GPU[$node]}
    used=$(squeue -w "$node" -h -o "%b" 2>/dev/null | grep -c "gpu" || true)
    free=$(( total - used ))
    w=$(( free - GPU_RESERVE ))
    [ "$w" -lt 0 ] && w=0
    [ "$w" -gt "$GPU_MAX_PER_NODE" ] && w=$GPU_MAX_PER_NODE
    NODE_WORKERS[$node]=$w
    echo "  ${node} (${NODE_PARTITION[$node]}): ${free}/${total} free  ->  ${w} workers"
    N_WORKERS=$(( N_WORKERS + w ))
done
echo "  Total workers: ${N_WORKERS}"
echo "==================================="

if [ "$N_WORKERS" -eq 0 ]; then
    echo "ERROR: No GPUs available. Try again later."
    exit 1
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

for node in "${SORTED_NODES[@]}"; do
    w=${NODE_WORKERS[$node]}
    [ "$w" -eq 0 ] && continue

    part=${NODE_PARTITION[$node]}
    start=$WORKER_OFFSET
    end=$(( WORKER_OFFSET + w - 1 ))

    job=$(sbatch --parsable \
        --partition="$part" \
        --qos="$QOS" \
        --nodelist="$node" \
        --array="${start}-${end}" \
        --export="$COMMON_EXPORT" \
        precompute_frame_dino.sbatch)
    JOB_IDS+=("$job")
    echo "${node} job: ${job}  (workers ${start}-${end})"

    WORKER_OFFSET=$(( WORKER_OFFSET + w ))
done

echo ""
echo "Submitted ${#JOB_IDS[@]} array job(s), total workers=${N_WORKERS}"
echo "Dataset : ${DATASET_ROOT}/${DATASET}"
echo "Backbone: ${VISUAL_BACKBONE}"
echo "Cameras : ${IMAGE_KEYS}"
echo "Output  : ${HOMEDIR}${PROJDIR}/outputs/${DATASET}_frame_dino_features"
echo "Monitor : squeue -u ${USER}"
echo "Logs    : tail -f logs/DP_frame_DINO_${JOB_IDS[0]}_0.err"
