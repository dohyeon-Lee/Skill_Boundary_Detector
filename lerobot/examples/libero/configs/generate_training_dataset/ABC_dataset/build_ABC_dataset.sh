#!/usr/bin/env bash
# Build the ABC LeRobot-v3 dataset(s) from the staged mcap subset:
#   ② mcap→abcdl (30Hz 리샘플+다운스케일 캐시)  ③ abcdl→v3 (pyav; torchcodec 불필요)  ④ quantile stats
# Download is NOT done here — run ./download_ABC.sh (or submit_build_ABC.sh) first.
#
# Usage:
#   ./build_ABC_dataset.sh                         # all subsets (이미 빌드된 것은 스킵)
#   ABC_ONLY="abc_toy" ./build_ABC_dataset.sh      # subset
#   FORCE=1 ./build_ABC_dataset.sh                 # 최종 v3 재빌드 (abcdl 캐시는 재사용)
#   WORKERS=8 ./build_ABC_dataset.sh               # mcap→abcdl 병렬도 오버라이드
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

# ── dep 사전 경고 (강제는 convert_abc_dataset.py가 진입점별로 — mcap 스테이징이 있을 때만
#    abcdl+mcap 계열이 필요; _abcdl 진입점(② 스킵)은 pyav+lerobot만 쓰므로 여기선 경고만) ──
"${BOOTSTRAP_PYTHON}" -c "import mcap, mcap_protobuf, foxglove_schemas_protobuf" 2>/dev/null || {
  echo "[deps] (경고) mcap 계열 미설치 — mcap→abcdl 단계가 필요하면 실행 전:" >&2
  echo '  uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf' >&2
}
"${BOOTSTRAP_PYTHON}" -c "import sys; sys.path.insert(0, '${ABCDL_REPO}'); import abcdl" 2>/dev/null || {
  echo "[deps] (경고) abcdl import 실패 (${ABCDL_REPO}) — mcap→abcdl 단계가 필요하면 yaml의 abcdl_repo 확인" >&2
}

ARGS=()
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "${WORKERS}")

echo "== ABC build (out: ${ABC_ROOT}) =="
ABC_ONLY="${ABC_ONLY:-}" FORCE="${FORCE:-}" \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/convert_abc_dataset.py" "${ARGS[@]}"
