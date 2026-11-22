#!/usr/bin/env bash
# SkillVLA remote dispatch (direct execution only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/../../../../remote.sh"
fi
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../../" && pwd)"
PYTHON="${PROJECT_ROOT}/.venv_abc_sim/bin/python"
RUN_DIR="${PROJECT_ROOT}/dataset_ABC/skillvla/abc_sim_all/FSQ333_contrastive_2000_pt"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ABC simulator environment is missing: ${PYTHON}" >&2
  echo "Run: bash setup_env.sh --abc-sim-only" >&2
  exit 1
fi

exec env PYTHONPATH="${PROJECT_ROOT}/third_party/abc:${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${SCRIPT_DIR}/src/preview_dual_focus_uv.py" \
  --skillvla-dir "${RUN_DIR}" "$@"
