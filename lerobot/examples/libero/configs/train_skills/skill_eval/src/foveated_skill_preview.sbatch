#!/usr/bin/env bash
#SBATCH --job-name=FOVEA_VIZ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

SUBMIT_DIR="${FOVEATED_PREVIEW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FOVEATED_PREVIEW_CONFIG:-${SUBMIT_DIR}/foveated_skill_preview_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/foveated_skill_preview_config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_CONFIG_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${LIBERO_CONFIG_TMP_ROOT}/fovea_libero.XXXXXX")"
trap 'rm -rf "${LIBERO_LOCAL_CONFIG_DIR}"' EXIT
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}" "${PREVIEW_OUTPUT_DIR}"
printf '%s\n' \
  "benchmark_root: ${LIBERO_BENCHMARK_ROOT}" \
  "bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files" \
  "init_states: ${LIBERO_BENCHMARK_ROOT}/init_files" \
  "datasets: ${LIBERO_SOURCE_ROOT}/datasets" \
  "assets: ${LIBERO_BENCHMARK_ROOT}/assets" \
  > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml"
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

for artifact in \
  "${SKILL_DATASET_DIR}" \
  "${SKILL_LATENTS_PATH}" \
  "${EVAL_INIT_STATES_PATH}" \
  "${ORIGINAL_DATASET_DIR}"; do
  if [ ! -e "${artifact}" ]; then
    echo "Missing foveated-preview artifact: ${artifact}" >&2
    exit 1
  fi
done

echo "START FOVEATED SKILL PREVIEW"
echo "node   : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "output : ${PREVIEW_OUTPUT_DIR}/index.html"
echo "CUDA   : not used (allocation only satisfies cluster QOS)"

PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_foveated_skill_preview.py" \
    --skill-dataset-dir "${SKILL_DATASET_DIR}" \
    --skill-latents-path "${SKILL_LATENTS_PATH}" \
    --eval-init-states-path "${EVAL_INIT_STATES_PATH}" \
    --original-dataset-dir "${ORIGINAL_DATASET_DIR}" \
    --target-task "${TARGET_TASK}" \
    --task-ids "${TASK_IDS}" \
    --episode-ids "${EPISODE_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --episode-selection "${EPISODE_SELECTION}" \
    --camera "${FOVEATION_CAMERA}" \
    --mode "${FOVEATION_MODE}" \
    --crop-size "${FOVEATION_CROP_SIZE}" \
    --output-size "${FOVEATION_OUTPUT_SIZE}" \
    --inner-box-enabled "${FOVEATION_INNER_BOX_ENABLED}" \
    --inner-box-mode "${FOVEATION_INNER_BOX_MODE}" \
    --inner-box-size "${FOVEATION_INNER_BOX_SIZE}" \
    --inner-box-line-width "${FOVEATION_INNER_BOX_LINE_WIDTH}" \
    --shape "${FOVEATION_SHAPE}" \
    --sharp-size "${FOVEATION_SHARP_SIZE}" \
    --feather "${FOVEATION_FEATHER}" \
    --blur-radius "${FOVEATION_BLUR_RADIUS}" \
    --random-color-enabled "${RANDOM_COLOR_ENABLED}" \
    --random-color-brightness "${RANDOM_COLOR_BRIGHTNESS}" \
    --random-color-contrast "${RANDOM_COLOR_CONTRAST}" \
    --random-color-saturation "${RANDOM_COLOR_SATURATION}" \
    --random-color-hue "${RANDOM_COLOR_HUE}" \
    --random-crop-enabled "${RANDOM_CROP_ENABLED}" \
    --random-crop-offset-px "${RANDOM_CROP_OFFSET_PX}" \
    --random-crop-inner-box-offset-px "${RANDOM_CROP_INNER_BOX_OFFSET_PX}" \
    --random-blur-enabled "${RANDOM_BLUR_ENABLED}" \
    --random-blur-radius "${RANDOM_BLUR_RADIUS}" \
    --output-dir "${PREVIEW_OUTPUT_DIR}" \
    --seed "${EVAL_SEED}"

echo "DONE -> ${PREVIEW_OUTPUT_DIR}/index.html"
