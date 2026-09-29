#!/usr/bin/env bash
#SBATCH --job-name=ATTN_VIZ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

SUBMIT_DIR="${ATTENTION_PREVIEW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${ATTENTION_PREVIEW_CONFIG:-${SUBMIT_DIR}/config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/config.py" \
    --config "${CONFIG_PATH}" --shell
)"
eval "${SETTINGS}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
unset LD_LIBRARY_PATH

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_CONFIG_TMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
LIBERO_LOCAL_CONFIG_DIR="$(mktemp -d "${LIBERO_CONFIG_TMP_ROOT}/attention_libero.XXXXXX")"
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
    echo "Missing attention-preview artifact: ${artifact}" >&2
    exit 1
  fi
done

echo "START SKILL ATTENTION-TARGET PREVIEW"
echo "node   : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "output : ${PREVIEW_OUTPUT_DIR}/index.html"
echo "device : CPU only"

PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/preview.py" \
    --skill-dataset-dir "${SKILL_DATASET_DIR}" \
    --skill-latents-path "${SKILL_LATENTS_PATH}" \
    --eval-init-states-path "${EVAL_INIT_STATES_PATH}" \
    --original-dataset-dir "${ORIGINAL_DATASET_DIR}" \
    --target-task "${TARGET_TASK}" \
    --task-ids "${TASK_IDS}" \
    --episode-ids "${EPISODE_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --episode-selection "${EPISODE_SELECTION}" \
    --agent-camera "${AGENT_CAMERA}" \
    --agent-video-key "${AGENT_VIDEO_KEY}" \
    --wrist-video-key "${WRIST_VIDEO_KEY}" \
    --patch-grid "${PATCH_GRID}" \
    --soft-sigma "${SOFT_SIGMA}" \
    --frames-per-skill "${FRAMES_PER_SKILL}" \
    --noise-std-m "${NOISE_STD_M}" \
    --noise-samples "${NOISE_SAMPLES}" \
    --output-dir "${PREVIEW_OUTPUT_DIR}" \
    --seed "${EVAL_SEED}"

echo "DONE -> ${PREVIEW_OUTPUT_DIR}/index.html"
