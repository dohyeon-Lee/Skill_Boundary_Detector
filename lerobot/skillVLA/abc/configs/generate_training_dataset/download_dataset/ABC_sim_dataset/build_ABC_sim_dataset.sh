#!/usr/bin/env bash
# Build a lightweight native view over downloaded official ABC Sim episodes.
# Download is a separate step; use download_ABC_sim.sh first.
#
# Usage:
#   ./build_ABC_sim_dataset.sh
#   FORCE=1 ./build_ABC_sim_dataset.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

export ABC_SIM_ONLY="${ABC_SIM_ONLY:-}"
export FORCE="${FORCE:-}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/build_abc_sim_native.py" "$@"
