#!/usr/bin/env bash
# Standalone scene capture: no policy checkpoint, inference or evaluation submission.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/tools/lerobot-libero${PYTHONPATH:+:${PYTHONPATH}}"
# Initial images need no model/GPU: software rendering also works on login/CPU nodes.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ "${MUJOCO_GL}" == osmesa || "${MUJOCO_GL}" == egl ]]; then
  export PYOPENGL_PLATFORM="${MUJOCO_GL}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}" "${SCRIPT_DIR}/src/capture_initial_frames.py" \
  --config "${PI05_EVAL_CONFIG:-${SCRIPT_DIR}/pi05_eval_config.yaml}" "$@"
