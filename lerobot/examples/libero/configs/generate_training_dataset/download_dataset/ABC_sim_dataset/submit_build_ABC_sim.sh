#!/usr/bin/env bash
# One-shot: download on the current networked node, then submit conversion to Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../src/submit_job.sh"
PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3
eval "$("${PYTHON}" "${SCRIPT_DIR}/src/ABC_sim_dataset_config.py" --shell)"

echo "== 1/2 ABC Sim download =="
ABC_SIM_ONLY="${ABC_SIM_ONLY:-}" "${SCRIPT_DIR}/download_ABC_sim.sh"

echo "== 2/2 submit LeRobot conversion =="
mkdir -p "${SCRIPT_DIR}/logs"
SBATCH_ARGS=(
  --job-name=build_ABC_sim
  --partition="${BUILD_PARTITION}"
  --qos="${BUILD_QOS}"
  --gres=gpu:1
  --cpus-per-task=32
  --mem=96G
  --time=48:00:00
  --output="${SCRIPT_DIR}/logs/%x_%j.out"
  --error="${SCRIPT_DIR}/logs/%x_%j.err"
)
if [ -n "${BUILD_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${BUILD_EXCLUDE_NODES}")
fi

WRAP="ABC_SIM_ONLY=$(printf %q "${ABC_SIM_ONLY:-}") FORCE=$(printf %q "${FORCE:-}") ${SCRIPT_DIR}/build_ABC_sim_dataset.sh"
bash "${_SUBMIT_JOB_SRC}/submit_job.sh" batch "${SBATCH_ARGS[@]}" --wrap="${WRAP}"

