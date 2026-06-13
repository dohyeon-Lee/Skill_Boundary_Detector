#!/usr/bin/env bash
# FSQ utilized eval — yaml의 모든 target에 대해 ①③(+옵션 ②) 실행 후 비교표 출력.
# 타깃·섹션별 결과는 outputs/{name}/metrics.json에 캐시 (이미 있으면 스킵, --force로 재계산).
# ②(z-swap)는 시작프레임 DINO 즉석 추론이라 CPU도 되지만 GPU에서 빠름 → submit_fsq_eval.sh 권장.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/fsq_utilized_eval_config.py" --shell)"

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

FORCE="${1:-}"
python "${SCRIPT_DIR}/src/eval_skill_space.py"   ${FORCE:+--force}
python "${SCRIPT_DIR}/src/eval_label_quality.py" ${FORCE:+--force}
python "${SCRIPT_DIR}/src/eval_zswap.py"         ${FORCE:+--force}
python "${SCRIPT_DIR}/src/report.py"
