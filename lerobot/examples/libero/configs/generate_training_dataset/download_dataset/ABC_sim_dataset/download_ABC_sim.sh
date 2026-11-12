#!/usr/bin/env bash
# Download official ABC Sim task tarballs selected by ABC_sim_dataset_config.yaml.
#
# Usage:
#   ./download_ABC_sim.sh --list-tasks
#   ./download_ABC_sim.sh --dry-run
#   ./download_ABC_sim.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

export ABC_SIM_ONLY="${ABC_SIM_ONLY:-}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/download_abc_sim.py" "$@"
