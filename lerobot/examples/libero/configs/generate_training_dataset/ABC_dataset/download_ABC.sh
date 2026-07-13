#!/usr/bin/env bash
# Download an ABC-130k mcap SUBSET (per ABC_dataset_config.yaml abc_subsets) into the
# staging area: {project_root}/{abc_dataset_root}/_mcap/{name}/data/{split}/...
# Conversion happens in build_ABC_dataset.sh (which does NOT re-download).
#
# Usage:
#   DRY_RUN=1 ./download_ABC.sh              # 태스크/에피소드 구조 + 계획만 출력 (다운로드 X)
#   ./download_ABC.sh                        # all subsets from the yaml
#   ABC_ONLY="abc_toy" ./download_ABC.sh     # subset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/ABC_dataset_config.py" --shell)"

# Robust transfer (filtered_dataset과 동일 컨벤션): rust 백엔드 + read timeout으로
# 죽은 소켓에 무한히 매달리지 않게. hf_transfer 없으면 순정 백엔드.
if "${BOOTSTRAP_PYTHON}" -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

echo "== ABC subset download (repo: ${ABC_HF_REPO}) =="
ABC_ONLY="${ABC_ONLY:-}" DRY_RUN="${DRY_RUN:-}" \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/download_abc_subset.py"
