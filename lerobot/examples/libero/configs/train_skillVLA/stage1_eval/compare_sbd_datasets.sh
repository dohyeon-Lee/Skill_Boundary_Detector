#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/sbd_dataset_compare_config.yaml}"

cd "${SCRIPT_DIR}/../../../../../.."
python "${SCRIPT_DIR}/src/compare_sbd_datasets.py" --config "${CONFIG}"
