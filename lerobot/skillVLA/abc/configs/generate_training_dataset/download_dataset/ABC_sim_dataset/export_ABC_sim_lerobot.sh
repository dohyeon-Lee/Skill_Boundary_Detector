#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../../remote.sh"
fi
# Optional compatibility export: materialize ABC Sim as a full LeRobot v3 dataset.
# Normal DP/FSQ/SkillVLA training should use build_ABC_sim_dataset.sh instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

export ABC_SIM_ONLY="${ABC_SIM_ONLY:-}"
export FORCE="${FORCE:-}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/convert_abc_sim_dataset.py" "$@"
