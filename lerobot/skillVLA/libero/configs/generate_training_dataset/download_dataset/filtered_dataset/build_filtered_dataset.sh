#!/usr/bin/env bash
# Build the FILTERED LIBERO datasets end-to-end. 최종 레이아웃:
#   {filtered_root}/{name}        ← 변환·remap·stats 끝난, 바로 쓰는 v3.0 데이터셋
#   {filtered_root}/_v21/{name}   ← 다운로드한 v2.1 원본 (보존; CLEAN_V21=1이면 삭제)
#
# 단계: ① download(v2.1) → ② convert v2.1→v3.0 → ③ 최종 위치로 이동 + 원본 복원
#       → ④ gripper remap(zero_close) → ⑤ quantile stats
# NOTE: upstream 변환기는 완료 시 root를 v3.0으로 "제자리 교체"하고 원본을 {root}_old로
#       옮긴다. 이 스크립트가 그 결과를 받아 v3.0은 final로, 원본은 _v21/{name}로 되돌린다.
# 이미 빌드된 suite( final 존재 )는 스킵 — 재실행 안전.
#
# Usage:
#   ./build_filtered_dataset.sh                                   # all suites
#   FILTERED_ONLY="libero_10_full_full" ./build_filtered_dataset.sh         # subset
#   RECOMPUTE_STATS=1 ./build_filtered_dataset.sh                 # stats를 데이터에서 재계산
#   CLEAN_V21=1 ./build_filtered_dataset.sh                       # 끝나면 v2.1 원본 삭제
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/filtered_dataset_config.py" --shell)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${SCRIPT_DIR}/download_filtered_libero.sh"

dataset_version() {  # codebase_version from meta/info.json ("" if missing)
  grep -m1 -o '"codebase_version"[^,]*' "$1/meta/info.json" 2>/dev/null | grep -o 'v[0-9][0-9.]*' || true
}

STAGING="${FILTERED_ROOT}/_v21"
for pair in ${FILTERED_SUITES}; do
  name="${pair%%=*}"
  if [ -n "${FILTERED_ONLY:-}" ] && ! grep -qw "${name}" <<<"${FILTERED_ONLY}"; then
    continue
  fi
  final="${FILTERED_ROOT}/${name}"
  v21="${STAGING}/${name}"
  old="${STAGING}/${name}_old"

  if [ -f "${final}/meta/info.json" ]; then
    echo "[skip] ${name}: already built (${final})"
    continue
  fi

  ver="$(dataset_version "${v21}")"
  case "${ver}" in
    v2.1)
      echo "== ${name}: ② v2.1 → v3.0 =="
      python -m lerobot.scripts.convert_dataset_v21_to_v30 \
        --repo-id "dohyeon/${name}" \
        --root "${v21}" \
        --push-to-hub false \
        --data-file-size-in-mb "${DATA_FILE_SIZE_IN_MB}" \
        --video-file-size-in-mb "${VIDEO_FILE_SIZE_IN_MB}"
      # 변환기 완료 후: v21 = v3.0 결과물, old = v2.1 원본
      ;;
    v3.0)
      echo "== ${name}: ② 스킵 (이전 실행이 변환을 이미 완료: ${v21} = v3.0) =="
      ;;
    *)
      echo "[error] ${name}: ${v21} 가 없거나 버전 인식 실패('${ver}') — 다운로드부터 확인" >&2
      exit 1
      ;;
  esac
  [ "$(dataset_version "${v21}")" = "v3.0" ] || { echo "[error] ${name}: 변환 후에도 v3.0이 아님" >&2; exit 1; }

  echo "== ${name}: ③ 최종 위치로 이동 + 원본 복원 =="
  mv "${v21}" "${final}"                       # 변환본 → dataset_filtered/{name}
  if [ -d "${old}" ]; then
    mv "${old}" "${v21}"                       # v2.1 원본 → _v21/{name} (보존)
  else
    echo "  (v2.1 백업 ${old} 없음 — 원본 보존 생략)"
  fi

  echo "== ${name}: ④ gripper remap (zero_close) =="
  python "${SCRIPT_DIR}/src/remap_gripper_zero_close.py" --dataset-dir "${final}"

  echo "== ${name}: ⑤ quantile stats =="
  if [ -n "${RECOMPUTE_STATS:-}" ]; then
    python "${SCRIPT_DIR}/src/ensure_quantile_stats.py" --dataset "${name}" --overwrite
  else
    python "${SCRIPT_DIR}/src/ensure_quantile_stats.py" --dataset "${name}"
  fi

  if [ -n "${CLEAN_V21:-}" ] && [ -d "${v21}" ]; then
    rm -rf "${v21}"
    echo "  (CLEAN_V21=1 → v2.1 원본 삭제)"
  fi
  echo "== ${name}: DONE → ${final} =="
done

echo "ALL DONE → ${FILTERED_ROOT}"
