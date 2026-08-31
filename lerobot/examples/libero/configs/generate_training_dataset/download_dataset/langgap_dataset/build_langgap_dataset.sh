#!/usr/bin/env bash
# Build the LangGap datasets end-to-end. 최종 레이아웃:
#   {langgap_root}/{name}        ← 20 Hz 재작성 끝난, 바로 쓰는 canonical v3.0 데이터셋
#   {langgap_root}/_hf/{name}    ← 다운로드한 HF 원본 (보존; CLEAN_HF=1이면 삭제)
#
# 단계: ① download(HF v3.0) → ② orientation verify(공통 task MSE 판정 + PNG)
#       → ③ full rewrite(20 Hz, image2→wrist_image, flip, gripper) → ④ quantile stats
# NOTE: ②의 verdict가 unknown이면 flip=auto인 ③이 안내와 함께 중단된다 —
#       PNG(_hf/{name}/.orientation/*_compare.png)를 눈으로 확인하고 yaml에 flip을 명시할 것.
# 이미 빌드된 세트( final 존재 )는 스킵 — 재실행 안전.
#
# Usage:
#   ./build_langgap_dataset.sh                                   # default sets
#   LANGGAP_ONLY="langgap_6_smoke" ./build_langgap_dataset.sh    # subset (incl. extra_sets)
#   RECOMPUTE_STATS=1 ./build_langgap_dataset.sh                 # stats를 데이터에서 재계산
#   CLEAN_HF=1 ./build_langgap_dataset.sh                        # 끝나면 HF 원본 삭제
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/langgap_dataset_config.py" --shell)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${SCRIPT_DIR}/download_langgap.sh"

STAGING="${LANGGAP_ROOT}/_hf"
TARGETS="${LANGGAP_ONLY:-${DEFAULT_SETS}}"

for pair in ${LANGGAP_SETS}; do
  name="${pair%%=*}"
  grep -qw "${name}" <<<"${TARGETS}" || continue
  final="${LANGGAP_ROOT}/${name}"
  hf="${STAGING}/${name}"

  if [ -f "${final}/meta/info.json" ]; then
    echo "[skip] ${name}: already built (${final})"
    continue
  fi

  echo "== ${name}: ② orientation verify =="
  python "${SCRIPT_DIR}/src/verify_image_orientation.py" --set "${name}"

  echo "== ${name}: ③ rewrite → canonical 20 Hz =="
  python "${SCRIPT_DIR}/src/convert_langgap_to_canonical.py" --set "${name}" \
    ${CONVERT_OVERWRITE:+--overwrite}

  echo "== ${name}: ④ quantile stats =="
  if [ -n "${RECOMPUTE_STATS:-}" ]; then
    python "${SCRIPT_DIR}/src/ensure_quantile_stats.py" --dataset "${name}" --overwrite
  else
    python "${SCRIPT_DIR}/src/ensure_quantile_stats.py" --dataset "${name}"
  fi

  if [ -n "${CLEAN_HF:-}" ] && [ -d "${hf}" ]; then
    rm -rf "${hf}"
    echo "  (CLEAN_HF=1 → HF 원본 삭제)"
  fi
  echo "== ${name}: DONE → ${final} =="
done

echo "ALL DONE → ${LANGGAP_ROOT}"
