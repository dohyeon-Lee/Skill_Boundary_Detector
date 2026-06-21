#!/usr/bin/env bash
# Side-by-side eval-video comparison. Edit video_compare_config.yaml (which runs to compare),
# then run this. Extra args are forwarded to compare_videos.py (override the yaml), e.g.:
#   ./run.sh                       # use the yaml as-is
#   ./run.sh --tasks 0 --episodes 0
#   ./run.sh np_state_c10 np_state_skill_c10
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# project_root = 7 levels up (…/stage1_eval/video_compare → SBD). Use its .venv (has imageio/cv2/PIL).
PROJECT_ROOT="$(cd "${HERE}/../../../../../../.." && pwd)"
PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || { echo "[error] project venv python not found: ${PY}" >&2; exit 1; }

exec "${PY}" "${HERE}/compare_videos.py" "$@"
