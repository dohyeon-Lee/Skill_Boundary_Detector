#!/usr/bin/env bash
#SBATCH --job-name=VSAeq
# VSA≡stage1 open-loop forward-equivalence test job (submitted by submit_verify.sh).
set -euo pipefail
# sbatch copies this script to /var/spool/slurmd → BASH_SOURCE dir is NOT the tools dir. Use TOOLS_DIR
# (exported by submit_verify.sh) to locate the python next to it.
HERE="${TOOLS_DIR:?TOOLS_DIR not set (submit via submit_verify.sh)}"

echo "START VSA≡stage1 verify"
echo "node   : ${SLURMD_NODENAME:-local}"
echo "stage2 : ${STAGE2_PATH}"
echo "stage1 : ${STAGE1_PATH}"
nvidia-smi || true

EXTRA=(--source "${SOURCE:-synthetic}")
if [ "${SOURCE:-synthetic}" = "dataset" ] && [ -n "${RAW_DATASET_DIR:-}" ]; then
  EXTRA+=(--raw_dataset_dir "${RAW_DATASET_DIR}")
fi
if [ "${ADAPTER_PROBES:-true}" = "true" ]; then
  EXTRA+=(--adapter_probes)
else
  EXTRA+=(--no-adapter-probes)
fi

mkdir -p "${RESULT_DIR:?RESULT_DIR not set}"
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero" \
  "${PROJECT_ROOT}/.venv/bin/python" "${HERE}/verify_vsa_stage1_equiv.py" \
    --stage2 "${STAGE2_PATH}" --stage1 "${STAGE1_PATH}" \
    --codes "${CODES:-0,31,62,93,124}" --num_steps "${NUM_STEPS:-10}" --device cuda \
    --dtype "${DTYPE:-asis}" \
    --out "${RESULT_DIR}/result.txt" \
    "${EXTRA[@]}"
