#!/usr/bin/env bash
# ONE-SHOT: download on THIS node (login → internet guaranteed), then submit the heavy
# part (mcap→abcdl→v3 conversion + stats) as a CPU Slurm job.
#
# Usage (from this folder):
#   ./submit_build_ABC.sh                              # all subsets
#   ABC_ONLY="abc_toy" ./submit_build_ABC.sh           # subset
#   FORCE=1 WORKERS=8 ./submit_build_ABC.sh            # knobs pass through
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

echo "== ① download (this node) =="
ABC_ONLY="${ABC_ONLY:-}" "${SCRIPT_DIR}/download_ABC.sh"

echo "== ② submit build job (mcap→abcdl→v3 + stats) =="
cd "${SCRIPT_DIR}"
mkdir -p logs

SBATCH_ARGS=(
  --job-name=build_ABC
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres=gpu:1            # CPU/IO 작업이지만 base_qos가 GPU>=1을 강제 (QOSMinGRES)
  --cpus-per-task=32      # ffmpeg 디코드/인코드 병렬 (convert_workers 16 × ~2 threads). 노드가
                          # 256코어라 여유. 스케줄 안 되면(qos cpu 상한) 16으로 낮출 것.
  --mem=96G
  --time=48:00:00         # ③(abcdl→v3)이 ~18-20h 실측 — v3는 재개 불가라 walltime 여유 필수
  --output=logs/%x_%j.out
  --error=logs/%x_%j.err
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

WRAP="ABC_ONLY=$(printf %q "${ABC_ONLY:-}") \
FORCE=$(printf %q "${FORCE:-}") \
WORKERS=$(printf %q "${WORKERS:-}") \
${SCRIPT_DIR}/build_ABC_dataset.sh"

echo "  subsets: ${ABC_ONLY:-(all)}"
echo "  out    : ${ABC_ROOT}"
sbatch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"
