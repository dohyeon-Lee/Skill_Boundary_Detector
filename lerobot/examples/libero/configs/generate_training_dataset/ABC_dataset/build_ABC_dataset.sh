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

# ── fail-fast dep checks (정확한 처방 출력) ──────────────────────────────────
"${BOOTSTRAP_PYTHON}" -c "import mcap, mcap_protobuf, foxglove_schemas_protobuf" 2>/dev/null || {
  echo "[deps] mcap 계열 미설치 — 아래 한 줄 실행 후 재시도:" >&2
  echo '  uv pip install mcap "mcap-protobuf-support>=0.5,<0.6" foxglove-schemas-protobuf' >&2
  exit 1
}
"${BOOTSTRAP_PYTHON}" -c "import sys; sys.path.insert(0, '${ABCDL_REPO}'); import abcdl" 2>/dev/null || {
  echo "[deps] abcdl import 실패 — yaml의 abcdl_repo(${ABCDL_REPO}) 확인" >&2
  exit 1
}

ARGS=()
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "${WORKERS}")

echo "== ABC build (out: ${ABC_ROOT}) =="
ABC_ONLY="${ABC_ONLY:-}" FORCE="${FORCE:-}" \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/convert_abc_dataset.py" "${ARGS[@]}"
