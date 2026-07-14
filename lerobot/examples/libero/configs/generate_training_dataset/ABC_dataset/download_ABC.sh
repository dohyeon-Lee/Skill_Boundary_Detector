#!/usr/bin/env bash
# Download ABC-130k mcap SUBSETs (per ABC_dataset_config.yaml abc_subsets) into the
# staging area: {project_root}/{abc_dataset_root}/_mcap/{name}/data/{split}/...
# 엔진 = {abcdl_repo}/download/src/download_abc.py (그룹/태스크/에피소드 선택형 다운로더;
# 논문 7개 primitive 카테고리 taxonomy 내장). Conversion happens in build_ABC_dataset.sh.
#
# NOTE: XDOF/ABC-130k는 gated — HF 페이지에서 라이선스 수락 + `huggingface-cli login` 선행.
#
# Usage:
#   ./download_ABC.sh --list-tasks           # 태스크 폴더명 전체 (--counts 로 에피소드 수까지)
#   ./download_ABC.sh --dry-run              # 계획만 출력 (다운로드 X)  (DRY_RUN=1 도 동일)
#   ./download_ABC.sh                        # all subsets from the yaml
#   ABC_ONLY="abc_toy" ./download_ABC.sh     # subset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

# 죽은 소켓에 무한히 매달리지 않게 (hf_transfer 자체는 엔진이 config로 켬).
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

echo "== ABC subset download (repo: ${ABC_HF_REPO}, engine: ${ABCDL_REPO}/download) =="
ABC_ONLY="${ABC_ONLY:-}" DRY_RUN="${DRY_RUN:-}" \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/download_abc_subset.py" "$@"
