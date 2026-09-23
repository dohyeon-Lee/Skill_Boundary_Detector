#!/usr/bin/env bash
# Draw skill-end EEF xyz on wrist-camera frames and write an HTML report.
#   ./run_wrist_uv_probe.sh [--config other.yaml]
# CPU only (a few decoded frames), so it runs here; no Slurm job needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${WRIST_UV_PROBE_CONFIG:-${SCRIPT_DIR}/wrist_uv_probe_config.yaml}"
if [ "${1:-}" = "--config" ]; then
  [ $# -ge 2 ] || { echo "--config requires a YAML path" >&2; exit 2; }
  CONFIG_PATH="$2"
  shift 2
fi
[ $# -eq 0 ] || { echo "Unexpected arguments: $*" >&2; exit 2; }

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PYTHON}" ] || { echo "Project venv Python not found: ${PYTHON}" >&2; exit 1; }

eval "$("${PYTHON}" "${SCRIPT_DIR}/src/wrist_uv_probe_config.py" --config "${CONFIG_PATH}" --shell)"

"${PYTHON}" "${SCRIPT_DIR}/src/wrist_uv_probe.py" \
  --skill-dataset-dir "${PROBE_SKILL_DATASET_DIR}" \
  --skill-latents-path "${PROBE_SKILL_LATENTS_PATH}" \
  --eval-init-states-path "${PROBE_EVAL_INIT_STATES_PATH}" \
  --original-dataset-dir "${PROBE_ORIGINAL_DATASET_DIR}" \
  --suite "${PROBE_SUITE}" \
  --camera "${PROBE_CAMERA}" \
  --eef-site "${PROBE_EEF_SITE}" \
  --rotation-frame "${PROBE_ROTATION_FRAME}" \
  --video-key "${PROBE_VIDEO_KEY}" \
  --task-ids "${PROBE_TASK_IDS}" \
  --episode-ids "${PROBE_EPISODE_IDS}" \
  --episodes "${PROBE_EPISODES}" \
  --skills-per-episode "${PROBE_SKILLS_PER_EPISODE}" \
  --frames-per-skill "${PROBE_FRAMES_PER_SKILL}" \
  --orientations "${PROBE_ORIENTATIONS}" \
  --patch-grid "${PROBE_PATCH_GRID}" \
  --soft-sigma "${PROBE_SOFT_SIGMA}" \
  --out "${PROBE_OUT_DIR}"
