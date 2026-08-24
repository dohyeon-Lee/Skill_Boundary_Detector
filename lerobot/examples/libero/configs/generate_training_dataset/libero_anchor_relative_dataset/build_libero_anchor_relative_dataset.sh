#!/usr/bin/env bash
# Build derived LIBERO datasets with absolute EEF commands plus relative-action stats.
#
# Usage:
#   ./build_libero_anchor_relative_dataset.sh
#   ANCHOR_RELATIVE_ONLY="libero_10_full_full_rel" ./build_libero_anchor_relative_dataset.sh
#   FORCE=1 MAX_EPISODES=2 ./build_libero_anchor_relative_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=${ANCHOR_RELATIVE_CONFIG:-${SCRIPT_DIR}/libero_anchor_relative_dataset_config.yaml}
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/libero_anchor_relative_dataset_config.py" --config "${CONFIG_PATH}" --shell)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(--config "${CONFIG_PATH}")
[ -n "${ANCHOR_RELATIVE_ONLY:-}" ] && ARGS+=(--only "${ANCHOR_RELATIVE_ONLY}")
if [ -n "${FORCE:-}" ] && [ "${FORCE}" != "0" ]; then
  if [ -n "${SHARD_INDEX:-}" ] && [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "  resume        : ignore FORCE after Slurm requeue (restart=${SLURM_RESTART_COUNT})"
  else
    ARGS+=(--overwrite)
  fi
fi
[ -n "${MAX_EPISODES:-}" ] && ARGS+=(--max-episodes "${MAX_EPISODES}")
[ -n "${SKIP_STATS:-}" ] && [ "${SKIP_STATS}" != "0" ] && ARGS+=(--skip-stats)
[ -n "${NUM_SHARDS:-}" ] && ARGS+=(--num-shards "${NUM_SHARDS}")
[ -n "${SHARD_INDEX:-}" ] && ARGS+=(--shard-index "${SHARD_INDEX}")
[ -n "${AGGREGATE_ONLY:-}" ] && [ "${AGGREGATE_ONLY}" != "0" ] && ARGS+=(--aggregate-only)

echo "== LIBERO absolute-EEF derived dataset build =="
echo "  datasets root : ${ANCHOR_RELATIVE_DATASET_ROOT}"
echo "  selected      : ${ANCHOR_RELATIVE_ONLY:-(all configured)}"
if [ -n "${SHARD_INDEX:-}" ]; then
  echo "  mode          : shard ${SHARD_INDEX}/${NUM_SHARDS}"
elif [ -n "${AGGREGATE_ONLY:-}" ] && [ "${AGGREGATE_ONLY}" != "0" ]; then
  echo "  mode          : aggregate ${NUM_SHARDS} shards + stats"
else
  echo "  mode          : direct"
fi
python "${SCRIPT_DIR}/src/build_libero_anchor_relative_dataset.py" "${ARGS[@]}"
