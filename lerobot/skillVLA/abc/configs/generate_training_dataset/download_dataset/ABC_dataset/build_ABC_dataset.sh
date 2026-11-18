#!/usr/bin/env bash
# Build the ABC LeRobot-v3 dataset(s) from the staged mcap subset:
#   ② MCAP→cache (30Hz 리샘플+다운스케일)  ③ cache→v3 (pyav)  ④ quantile stats
# Download is NOT done here — run ./download_ABC.sh (or submit_build_ABC.sh) first.
#
# Usage:
#   ./build_ABC_dataset.sh                         # all subsets (이미 빌드된 것은 스킵)
#   ABC_ONLY="abc_toy" ./build_ABC_dataset.sh      # subset
#   FORCE=1 ./build_ABC_dataset.sh                 # 최종 v3 재빌드 (cache는 재사용)
#   WORKERS=8 ./build_ABC_dataset.sh               # MCAP→cache 병렬도 오버라이드
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

# ── dep 사전 경고 (MCAP 스테이징이 있을 때만 필요; cache 진입점은 불필요) ──
"${BOOTSTRAP_PYTHON}" -c "import mcap, mcap_protobuf, foxglove_schemas_protobuf" 2>/dev/null || {
  echo "[deps] (경고) ABC MCAP 의존성 미설치 — 저장소 루트에서 실행:" >&2
  echo '  bash setup_env.sh' >&2
}
ARGS=()
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "${WORKERS}")

echo "== ABC build (out: ${ABC_ROOT}) =="
ABC_ONLY="${ABC_ONLY:-}" FORCE="${FORCE:-}" \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/convert_abc_dataset.py" "${ARGS[@]}"
