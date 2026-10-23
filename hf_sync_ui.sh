#!/usr/bin/env bash
# Run on the server whose files you want to sync; forward port 8766 to your browser.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${HF_SYNC_UI_PYTHON:-}" ]; then
    HF_SYNC_PYTHON="${HF_SYNC_UI_PYTHON}"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    HF_SYNC_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    HF_SYNC_PYTHON=python3
fi
"${HF_SYNC_PYTHON}" -c 'import sys; sys.exit("Python 3.10+ is required; use HF_SYNC_UI_PYTHON=/path/to/python") if sys.version_info < (3, 10) else None'
exec "${HF_SYNC_PYTHON}" "${SCRIPT_DIR}/tools/server_sync/hf_server.py" "$@"
