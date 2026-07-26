#!/usr/bin/env bash
# ONE-SHOT: download + orientation verify on THIS node (login → internet guaranteed),
# then submit the heavy part (full rewrite + stats) as a Slurm job.
# (재작성은 전 프레임 비디오 디코드+AV1 재인코딩이라 filtered보다 무겁다 — original 변환급 리소스.)
#
# Usage (from this folder):
#   ./submit_build_langgap.sh                                   # default sets
#   LANGGAP_ONLY="langgap_6_smoke" ./submit_build_langgap.sh    # subset
#   RECOMPUTE_STATS=1 CLEAN_HF=1 ./submit_build_langgap.sh      # knobs pass through
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/langgap_dataset_config.py" --shell)"

echo "== ① download (this node) =="
LANGGAP_ONLY="${LANGGAP_ONLY:-}" "${SCRIPT_DIR}/download_langgap.sh"

echo "== ② orientation verify (this node) =="
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TARGETS="${LANGGAP_ONLY:-${DEFAULT_SETS}}"
for pair in ${LANGGAP_SETS}; do
  name="${pair%%=*}"
  grep -qw "${name}" <<<"${TARGETS}" || continue
  python "${SCRIPT_DIR}/verify_image_orientation.py" --set "${name}"
done

echo "== ③ submit build job (rewrite + stats) =="
cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=build_langgap
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres="${CONVERT_GRES}"
  --cpus-per-task="${CONVERT_CPUS_PER_TASK}"
  --mem="${CONVERT_MEM}"
  --time="${CONVERT_TIME}"
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

WRAP="LANGGAP_ONLY=$(printf %q "${LANGGAP_ONLY:-}") \
RECOMPUTE_STATS=$(printf %q "${RECOMPUTE_STATS:-}") \
CLEAN_HF=$(printf %q "${CLEAN_HF:-}") \
${SCRIPT_DIR}/build_langgap_dataset.sh"

echo "  sets   : ${LANGGAP_ONLY:-(default: ${DEFAULT_SETS})}"
echo "  out    : ${LANGGAP_ROOT}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
