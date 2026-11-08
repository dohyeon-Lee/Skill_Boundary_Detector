#!/usr/bin/env bash
# Run on the external server. Open through VSCode port forwarding / SSH -L.
set -euo pipefail
# Connect directly; inherited shell/Codex proxies may reject Hugging Face.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SYNC_UI_PYTHON:-}" ]; then
    SYNC_PYTHON="${SYNC_UI_PYTHON}"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    SYNC_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    SYNC_PYTHON=python3
fi
"${SYNC_PYTHON}" -c 'import sys; sys.exit("Python 3.10+ is required; use SYNC_UI_PYTHON=/path/to/python") if sys.version_info < (3, 10) else None'
exec "${SYNC_PYTHON}" "${SCRIPT_DIR}/tools/server_sync/desk.py" "$@"
