#!/usr/bin/env bash
# Compatibility entrypoint. All management tabs now share Sync Desk (port 8765).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_MONITOR_PYTHON:-}" ] && [ -z "${SYNC_UI_PYTHON:-}" ]; then
    export SYNC_UI_PYTHON="${SLURM_MONITOR_PYTHON}"
fi
exec bash "${SCRIPT_DIR}/sync_server_ui.sh" "$@"
