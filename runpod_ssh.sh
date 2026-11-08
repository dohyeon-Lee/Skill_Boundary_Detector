#!/usr/bin/env bash
# Run on the PC that starts SSH / VS Code connections.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# No training environment is required: prefer an installed Python 3.9+.
PYTHON_CMD=()
for candidate in python3 python "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/.venv/Scripts/python.exe"; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
        PYTHON_CMD=("$candidate")
        break
    fi
done
if [ "${#PYTHON_CMD[@]}" -eq 0 ] && command -v py >/dev/null 2>&1 &&
    py -3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
fi
if [ "${#PYTHON_CMD[@]}" -eq 0 ]; then
    echo "ERROR: Python 3.9 이상을 설치한 뒤 다시 실행하세요. setup_env.sh 설치는 필요하지 않습니다." >&2
    exit 1
fi
exec "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/tools/runpod/update_ssh_config.py" "$@"
