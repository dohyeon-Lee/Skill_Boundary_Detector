#!/usr/bin/env bash
# Serve a skill-eval report for human review.
#   ./review.sh                       # newest result with a merged manifest
#   ./review.sh <result-folder-name>  # specific result under outputs/
#   ./review.sh <result> --port 9000  # extra args pass through to review_server.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${SCRIPT_DIR}/outputs"
PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

TARGET=""
if [ $# -gt 0 ] && [[ "${1}" != -* ]]; then
  TARGET="${1}"
  shift
fi

if [ -z "${TARGET}" ]; then
  for d in $(ls -1dt "${OUT_ROOT}"/*/ 2>/dev/null); do
    if [ -f "${d}metrics/manifest.json" ]; then
      TARGET="$(basename "${d}")"
      break
    fi
  done
  if [ -z "${TARGET}" ]; then
    echo "No result with metrics/manifest.json under ${OUT_ROOT}" >&2
    exit 1
  fi
  echo "Auto-selected result: ${TARGET}"
fi

if [ -d "${TARGET}" ]; then
  TARGET_DIR="${TARGET}"
else
  TARGET_DIR="${OUT_ROOT}/${TARGET}"
fi

if [ ! -f "${TARGET_DIR}/metrics/manifest.json" ]; then
  echo "metrics/manifest.json not found in ${TARGET_DIR}" >&2
  echo "Reviewable results:" >&2
  for d in $(ls -1dt "${OUT_ROOT}"/*/ 2>/dev/null); do
    [ -f "${d}metrics/manifest.json" ] && echo "  $(basename "${d}")" >&2
  done
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/src/review_server.py" "${TARGET_DIR}" --refresh-html "$@"
