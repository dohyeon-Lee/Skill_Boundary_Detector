#!/usr/bin/env bash
# Run in tmux on the Slurm login server; forward port 8767 to your browser.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_MONITOR_PYTHON:-}" ]; then
    MONITOR_PYTHON="${SLURM_MONITOR_PYTHON}"
elif [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    MONITOR_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    MONITOR_PYTHON=python3
fi
"${MONITOR_PYTHON}" -c 'import sys; sys.exit("Python 3.10+ is required; set SLURM_MONITOR_PYTHON=/path/to/python") if sys.version_info < (3, 10) else None'
exec "${MONITOR_PYTHON}" "${SCRIPT_DIR}/tools/slurm_monitor/server.py" "$@"
