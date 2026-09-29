#!/usr/bin/env bash
# Submit the codebook-linked dual-camera attention-target preview.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${ATTENTION_PREVIEW_CONFIG:-${SCRIPT_DIR}/config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/src/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/src/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"

mkdir -p "${SCRIPT_DIR}/logs" "${PREVIEW_OUTPUT_DIR}"
SBATCH_ARGS=(
  --job-name=ATTN_VIZ
  --partition="${PREVIEW_PARTITION}"
  --qos="${PREVIEW_QOS}"
  --cpus-per-task="${PREVIEW_CPUS}"
  --mem="${PREVIEW_MEMORY}"
  --time="${PREVIEW_TIME}"
  --output="${SCRIPT_DIR}/logs/%x_%j.out"
  --error="${SCRIPT_DIR}/logs/%x_%j.err"
)
[ -z "${PREVIEW_GRES}" ] || SBATCH_ARGS+=(--gres="${PREVIEW_GRES}")
[ -z "${PREVIEW_NODELIST}" ] || SBATCH_ARGS+=(--nodelist="${PREVIEW_NODELIST}")
[ -z "${PREVIEW_EXCLUDE_NODES}" ] || SBATCH_ARGS+=(--exclude="${PREVIEW_EXCLUDE_NODES}")

echo "Submit skill attention-target preview"
echo "  dataset : ${SKILL_DATASET_DIR}"
echo "  skills  : ${SKILL_LATENTS_PATH}"
echo "  tasks   : ${TARGET_TASK} ${TASK_IDS}"
echo "  cameras : ${AGENT_VIDEO_KEY} + ${WRIST_VIDEO_KEY}"
echo "  target  : ${PATCH_GRID}x${PATCH_GRID} patches, sigma=${SOFT_SIGMA}, frames/skill=${FRAMES_PER_SKILL}"
echo "  noise   : std(m)=${NOISE_STD_M}, samples=${NOISE_SAMPLES} (input only)"
echo "  output  : ${PREVIEW_OUTPUT_DIR}/index.html"

ATTENTION_PREVIEW_DIR="${SCRIPT_DIR}" ATTENTION_PREVIEW_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/run.sbatch"
