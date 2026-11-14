#!/usr/bin/env bash
# CPU-only visualization of the already-built LIBERO skillset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${LIBERO_SBD_REPORT_CONFIG:-${SCRIPT_DIR}/config.yaml}"
BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

source "${SCRIPT_DIR}/../../../src/submit_job.sh"
if ! RESOLVED="$(
  "${BOOTSTRAP_PYTHON}" "${SCRIPT_DIR}/src/config.py" \
    --config "${CONFIG}" --shell
)"; then
  echo "LIBERO SBD report bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED}"

[ -f "${DP_ABLATION_SKILLSET_DIR}/skillset_manifest.json" ] || {
  echo "LIBERO skillset is incomplete: ${DP_ABLATION_SKILLSET_DIR}" >&2
  exit 1
}

cd "${SCRIPT_DIR}"
mkdir -p logs "${DP_BUILT_REPORT_OUTPUT_DIR}"
SBATCH_ARGS=(
  --job-name=LIBERO_SBD_report
  --partition="${DP_BUILT_REPORT_CPU_PARTITION}"
  --qos="${DP_BUILT_REPORT_CPU_QOS}"
  --cpus-per-task=8
  --mem=32G
  --time=02:00:00
  --export="ALL,LIBERO_SBD_REPORT_DIR=${SCRIPT_DIR},LIBERO_SBD_REPORT_CONFIG=${CONFIG}"
)
[ -n "${DP_BUILT_REPORT_CPU_ACCOUNT}" ] && SBATCH_ARGS+=(--account="${DP_BUILT_REPORT_CPU_ACCOUNT}")

echo "Submit LIBERO built-SBD visualization"
echo "  skillset : ${DP_ABLATION_SKILLSET_DIR}"
echo "  tasks    : ${DP_BUILT_REPORT_TASK_IDS} (${DP_BUILT_REPORT_TASK_ID_SPACE})"
echo "  episodes : ${DP_BUILT_REPORT_N_EPISODES} per task"
echo "  resource : CPU only (no DP inference)"
echo "  report   : ${DP_BUILT_REPORT_OUTPUT_DIR}/index.html"
bash "${_SUBMIT_JOB_SRC}/submit_job.sh" batch \
  "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/src/run_report.sbatch"
