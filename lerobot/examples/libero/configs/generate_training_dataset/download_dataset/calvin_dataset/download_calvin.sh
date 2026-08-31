#!/usr/bin/env bash
# Download, verify, and extract official raw CALVIN play-data variants.
# Task-disjoint filtering and LeRobot conversion intentionally belong to the later build step.
#
# Usage:
#   ./download_calvin.sh
#   CALVIN_ONLY="D" ./download_calvin.sh
#   ./download_calvin.sh --list-variants
#   ./download_calvin.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CALVIN_CONFIG:-${SCRIPT_DIR}/calvin_dataset_config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if ! SHELL_SETTINGS="$(PYTHONDONTWRITEBYTECODE=1 "${BOOTSTRAP_PYTHON}" \
  "${SCRIPT_DIR}/src/calvin_dataset_config.py" --config "${CONFIG_PATH}" --shell)"; then
  exit 1
fi
eval "${SHELL_SETTINGS}"

echo "== CALVIN raw download =="
echo "  variants: ${CALVIN_DOWNLOAD_VARIANTS}"
echo "  staging : ${CALVIN_RAW_ROOT}"

CALVIN_ONLY="${CALVIN_ONLY:-}" \
  PYTHONDONTWRITEBYTECODE=1 \
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/download_calvin.py" \
  --config "${CONFIG_PATH}" "$@"
