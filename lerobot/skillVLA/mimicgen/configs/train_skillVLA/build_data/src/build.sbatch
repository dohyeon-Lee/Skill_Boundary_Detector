#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "${MIMICGEN_BUILD_DIR:?}/../../../../../.." && pwd)"
export PYTHONPATH="${PROJECT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PROJECT}/.venv/bin/python" -u "${MIMICGEN_BUILD_DIR}/src/build_dataset.py" --config "${MIMICGEN_BUILD_CONFIG:?}"
