#!/usr/bin/env bash
# Default SBD evaluation: visualize the boundaries already produced by build_data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_CONFIG="${ABC_BUILD_DATA_CONFIG:-${SCRIPT_DIR}/../../build_data/build_data_config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

source "${SCRIPT_DIR}/../../../src/submit_job.sh"
RESOLVED="$("${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/../../src/train_skills_config.py" \
  --config "${BUILD_CONFIG}" --stage build_data --shell)"
eval "${RESOLVED}"
[ -f "${SKILLSET_DONE_PATH}" ] || {
  echo "ABC skillset is incomplete: ${SKILLSET_DONE_PATH}" >&2
  exit 1
}

REPORT_EPISODES="${ABC_SBD_REPORT_EPISODES:-5}"
case "${REPORT_EPISODES}" in
  ''|*[!0-9]*|0) echo "ABC_SBD_REPORT_EPISODES must be a positive integer." >&2; exit 2 ;;
esac
BOUNDARY_SOURCE="${ABC_SBD_BOUNDARY_SOURCE:-skillset}"
case "${BOUNDARY_SOURCE}" in
  skillset)
    REPORT_VARIANT="built_ck${DP_CHECKPOINT}"
    ;;
  denoising_gain)
    REPORT_VARIANT="denoising_gain_globalmean_ck${DP_CHECKPOINT}"
    ;;
  all_peaks)
    REPORT_VARIANT="all_peaks_ck${DP_CHECKPOINT}"
    ;;
  power_half_mean)
    REPORT_VARIANT="power0p5_globalmean_ck${DP_CHECKPOINT}"
    ;;
  power_one_initial_motion_mask)
    REPORT_VARIANT="power1_initial_motion_mask100_ck${DP_CHECKPOINT}"
    ;;
  *)
    echo "ABC_SBD_BOUNDARY_SOURCE must be skillset, denoising_gain, all_peaks, power_half_mean, or power_one_initial_motion_mask." >&2
    exit 2
    ;;
esac
REPORT_DIR="${PROJECT_ROOT}/lerobot/examples/analysis/skill_boundary_detection/abc/${TARGET_DATASET}/${REPORT_VARIANT}"

cd "${SCRIPT_DIR}"
mkdir -p logs "${REPORT_DIR}"
SBATCH_ARGS=(
  --job-name=ABC_SBD_report
  --partition="${SKILLSET_CPU_PARTITION}"
  --qos="${SKILLSET_CPU_QOS}"
  --cpus-per-task=8
  --mem=32G
  --time=02:00:00
  --export="ALL,ABC_SBD_REPORT_DIR=${SCRIPT_DIR},ABC_BUILD_DATA_CONFIG=${BUILD_CONFIG},ABC_SBD_REPORT_EPISODES=${REPORT_EPISODES},ABC_SBD_BOUNDARY_SOURCE=${BOUNDARY_SOURCE},ABC_SBD_REPORT_OUTPUT=${REPORT_DIR}"
)

echo "Submit ABC SBD build-result visualization"
echo "  skillset : ${SKILLSET_DIR}"
echo "  boundary : ${BOUNDARY_SOURCE}"
echo "  episodes : ${REPORT_EPISODES} per language prompt (${ABC_TASK_COUNT} prompts)"
echo "  resource : CPU only (no DP inference)"
echo "  report   : ${REPORT_DIR}/index.html"
submit_job "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/run_report.sbatch"
