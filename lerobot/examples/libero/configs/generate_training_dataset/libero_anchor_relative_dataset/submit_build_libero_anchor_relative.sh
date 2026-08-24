#!/usr/bin/env bash
# Submit one balanced episode array per dataset, followed by an afterok aggregate job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=${ANCHOR_RELATIVE_CONFIG:-${SCRIPT_DIR}/libero_anchor_relative_dataset_config.yaml}
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/libero_anchor_relative_dataset_config.py" --config "${CONFIG_PATH}" --shell)"

if [ "${CONVERT_NUM_SHARDS}" -le 0 ]; then
  echo "convert_num_shards must be positive, got ${CONVERT_NUM_SHARDS}" >&2
  exit 2
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

if [ "${DRY_RUN:-0}" = 0 ] && [ "${ALLOW_CONCURRENT:-0}" = 0 ]; then
  ACTIVE_BUILDERS=$(squeue -h -u "${USER}" -o '%A|%j|%T|%r' 2>/dev/null \
    | awk -F'|' '($2 == "build_libero_eef_relative" || $2 ~ /^librel_/) \
      && $4 !~ /DependencyNeverSatisfied/ {print $1 ":" $2 ":" $3}' || true)
  if [ -n "${ACTIVE_BUILDERS}" ]; then
    echo "An existing LIBERO relative builder is active (${ACTIVE_BUILDERS})." >&2
    echo "Wait/cancel it first, or set ALLOW_CONCURRENT=1 only if output paths cannot overlap." >&2
    exit 2
  fi
fi

REQUESTED=${ANCHOR_RELATIVE_ONLY:-${ANCHOR_RELATIVE_DATASET_NAMES}}
REQUESTED=${REQUESTED//,/ }
read -r -a DATASETS <<< "${REQUESTED}"
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "No datasets selected" >&2
  exit 2
fi
for dataset in "${DATASETS[@]}"; do
  case " ${ANCHOR_RELATIVE_DATASET_NAMES} " in
    *" ${dataset} "*) ;;
    *) echo "Unknown configured dataset: ${dataset}" >&2; exit 2 ;;
  esac
done

SBATCH_COMMON=(
  --requeue
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres="${CONVERT_GRES}"
  --cpus-per-task="${CONVERT_CPUS_PER_TASK}"
  --mem="${CONVERT_MEM}"
  --time="${CONVERT_TIME}"
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_COMMON+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

submit_sbatch() {
  if [ "${DRY_RUN:-0}" != 0 ]; then
    printf 'DRY RUN: sbatch' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    echo "dry-run-job"
  else
    sbatch --parsable "$@"
  fi
}

echo "  datasets : ${DATASETS[*]}"
echo "  output   : ${ANCHOR_RELATIVE_DATASET_ROOT}"
echo "  shards   : ${CONVERT_NUM_SHARDS} per dataset"
for dataset in "${DATASETS[@]}"; do
  job_tag=${dataset//[^a-zA-Z0-9]/_}
  job_tag=${job_tag:0:24}
  SHARD_WRAP="ANCHOR_RELATIVE_CONFIG=$(printf %q "${CONFIG_PATH}") \
ANCHOR_RELATIVE_ONLY=$(printf %q "${dataset}") \
NUM_SHARDS=$(printf %q "${CONVERT_NUM_SHARDS}") \
SHARD_INDEX=\${SLURM_ARRAY_TASK_ID} \
FORCE=$(printf %q "${RESET_SHARDS:-}") \
MAX_EPISODES=$(printf %q "${MAX_EPISODES:-}") \
SKIP_STATS=1 \
${SCRIPT_DIR}/build_libero_anchor_relative_dataset.sh"
  shard_job=$(submit_sbatch \
    "${SBATCH_COMMON[@]}" \
    --job-name="librel_${job_tag}_shard" \
    --array="0-$((CONVERT_NUM_SHARDS - 1))" \
    --output='logs/%x_%A_%a.out' \
    --error='logs/%x_%A_%a.err' \
    --wrap="${SHARD_WRAP}")
  shard_job=${shard_job%%;*}

  AGGREGATE_WRAP="ANCHOR_RELATIVE_CONFIG=$(printf %q "${CONFIG_PATH}") \
ANCHOR_RELATIVE_ONLY=$(printf %q "${dataset}") \
NUM_SHARDS=$(printf %q "${CONVERT_NUM_SHARDS}") \
AGGREGATE_ONLY=1 \
FORCE=$(printf %q "${FORCE:-}") \
SKIP_STATS=$(printf %q "${SKIP_STATS:-}") \
${SCRIPT_DIR}/build_libero_anchor_relative_dataset.sh"
  aggregate_job=$(submit_sbatch \
    "${SBATCH_COMMON[@]}" \
    --dependency="afterok:${shard_job}" \
    --job-name="librel_${job_tag}_merge" \
    --output='logs/%x_%j.out' \
    --error='logs/%x_%j.err' \
    --wrap="${AGGREGATE_WRAP}")
  aggregate_job=${aggregate_job%%;*}
  echo "  ${dataset}: shard array=${shard_job}, aggregate=${aggregate_job} (afterok)"
done
