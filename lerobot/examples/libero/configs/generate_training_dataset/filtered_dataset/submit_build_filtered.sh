#!/usr/bin/env bash
# ONE-SHOT: download on THIS node (login → internet guaranteed), then submit the heavy
# part (v2.1→v3.0 conversion + gripper remap + stats) as a CPU Slurm job.
#
# Usage (from this folder):
#   ./submit_build_filtered.sh                                  # all suites
#   FILTERED_ONLY="libero_90" ./submit_build_filtered.sh        # subset
#   RECOMPUTE_STATS=1 CLEAN_V21=1 ./submit_build_filtered.sh    # knobs pass through
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/filtered_dataset_config.py" --shell)"

echo "== ① download (this node) =="
FILTERED_ONLY="${FILTERED_ONLY:-}" "${SCRIPT_DIR}/download_filtered_libero.sh"

echo "== ② submit build job (convert + remap + stats) =="
cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=build_filtered
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres=gpu:1            # CPU/IO 작업이지만 base_qos가 GPU>=1을 강제 (QOSMinGRES)
  --cpus-per-task=8
  --mem=32G
  --time=12:00:00
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

WRAP="FILTERED_ONLY=$(printf %q "${FILTERED_ONLY:-}") \
RECOMPUTE_STATS=$(printf %q "${RECOMPUTE_STATS:-}") \
CLEAN_V21=$(printf %q "${CLEAN_V21:-}") \
${SCRIPT_DIR}/build_filtered_dataset.sh"

echo "  suites : ${FILTERED_ONLY:-(all)}"
echo "  out    : ${FILTERED_ROOT}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
