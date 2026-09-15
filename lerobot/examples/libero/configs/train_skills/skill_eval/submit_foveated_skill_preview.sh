#!/usr/bin/env bash
# Submit the codebook-linked foveated top-image preview.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_PATH="${FOVEATED_PREVIEW_CONFIG:-${SCRIPT_DIR}/foveated_skill_preview_config.yaml}"

CONFIG_LIB="$(dirname "${CONFIG_PATH}")"
while [ ! -f "${CONFIG_LIB}/snapshot_config.sh" ]; do CONFIG_LIB="$(dirname "${CONFIG_LIB}")"; done
source "${CONFIG_LIB}/snapshot_config.sh"
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/foveated_skill_preview_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"

mkdir -p "${SCRIPT_DIR}/logs" "${PREVIEW_OUTPUT_DIR}"
SBATCH_ARGS=(
  --job-name=FOVEA_VIZ
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

echo "Submit foveated skill preview"
echo "  dataset : ${SKILL_DATASET_DIR}"
echo "  skills  : ${SKILL_LATENTS_PATH}"
echo "  tasks   : ${TARGET_TASK} ${TASK_IDS}"
if [ "${FOVEATION_MODE}" = "crop" ]; then
  echo "  focus   : crop ${FOVEATION_CROP_SIZE}px -> ${FOVEATION_OUTPUT_SIZE}px, inner=${FOVEATION_INNER_BOX_MODE}:${FOVEATION_INNER_BOX_SIZE}px"
else
  echo "  focus   : partial_fov ${FOVEATION_SHAPE} sharp=${FOVEATION_SHARP_SIZE}px feather=${FOVEATION_FEATHER}px blur=${FOVEATION_BLUR_RADIUS}"
fi
echo "  random  : color=${RANDOM_COLOR_ENABLED} crop=${RANDOM_CROP_ENABLED} outer=${RANDOM_CROP_OFFSET_PX} red=${RANDOM_CROP_INNER_BOX_OFFSET_PX} blur=${RANDOM_BLUR_ENABLED}"
echo "  output  : ${PREVIEW_OUTPUT_DIR}/index.html"

FOVEATED_PREVIEW_DIR="${SCRIPT_DIR}" FOVEATED_PREVIEW_CONFIG="${CONFIG_PATH}" \
  sbatch "${SBATCH_ARGS[@]}" "${SRC_DIR}/foveated_skill_preview.sbatch"
