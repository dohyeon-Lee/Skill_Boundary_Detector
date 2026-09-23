#!/usr/bin/env bash
# Inputs:
#   config : ./dp_eval_config.yaml  (roots + DP selection + HTML knobs + slurm)
# Outputs:
#   DP : ./outputs/dp_skillset/{dataset}/{output_suffix}/index.html
#
# Run the CPU-only DP skill-boundary eval directly on the current host (boxed
# start/end frames per skill + multimodality curves). FSQ reconstruction still
# uses submit_fsq_eval.sh and Slurm/GPU independently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}}"

# This script runs exactly the DP eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_DP=true
export EVAL_RUN_FSQ=false

# Freeze the config so this job ignores later edits to the repo yaml (see configs/src/snapshot_config.sh).
_lib="$(dirname "${EVAL_CONFIG}")"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
EVAL_CONFIG="$(snapshot_config "${EVAL_CONFIG}")"

BOOTSTRAP_PYTHON="${SCRIPT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Evaluation-only knobs, selected artifact, global roots, and Slurm resources.
# Capture first and eval second: `eval "$(failed command)"` returns success for
# empty output and would otherwise allow an invalid job to be submitted.
if ! RESOLVED_EVAL="$(
  "${BOOTSTRAP_PYTHON}" "${EVAL_SRC_DIR}/eval_config.py" \
    --config "${EVAL_CONFIG}" --shell
)"; then
  echo "DP evaluation bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${RESOLVED_EVAL}"

# Every selected artifact must already exist. Validate the complete list before
# submitting so a typo cannot consume a queued GPU job.
if ! DP_VALIDATION="$("${BOOTSTRAP_PYTHON}" - "${DP_EVAL_SKILLSETS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

specs = json.loads(sys.argv[1])
if not isinstance(specs, list) or not specs:
    raise ValueError("dp_eval_skillsets is empty")
dataset_names = set()
for spec in specs:
    skillset_dir = Path(spec["skillset_dir"])
    manifest_path = skillset_dir / "skillset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"skillset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    dataset_name = str(manifest.get("dataset_name", "")).strip()
    dataset_dir = Path(str(manifest.get("dataset_dir", "")).strip())
    policy_raw = str(manifest.get("policy_path", "")).strip()
    if not dataset_name:
        raise ValueError(f"manifest dataset_name is empty: {manifest_path}")
    if not policy_raw or len(Path(policy_raw).parents) < 3:
        raise ValueError(f"invalid manifest policy_path: {policy_raw!r}")
    if not (skillset_dir / "skills").is_dir():
        raise FileNotFoundError(f"skills directory not found: {skillset_dir / 'skills'}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"source dataset videos not found: {dataset_dir / 'videos'}")
    dataset_names.add(dataset_name)
if len(dataset_names) != 1:
    raise ValueError(f"all comparison artifacts must use one dataset, got {sorted(dataset_names)}")
print(next(iter(dataset_names)))
print(len(specs))
print(", ".join(str(spec["label"]) for spec in specs))
PY
  )"; then
  echo "Invalid DP comparison artifact list." >&2
  exit 1
fi
mapfile -t DP_VALIDATION_LINES <<< "${DP_VALIDATION}"
TARGET_DATASET="${DP_VALIDATION_LINES[0]}"
DP_MODEL_COUNT="${DP_VALIDATION_LINES[1]}"
DP_MODEL_LABELS="${DP_VALIDATION_LINES[2]}"

cd "${SCRIPT_DIR}"
mkdir -p logs outputs

echo "Run DP skill-boundary eval locally (CPU only)"
echo "  models      : ${DP_MODEL_COUNT} (${DP_MODEL_LABELS})"
echo "  dataset     : ${TARGET_DATASET} (from manifest)"
echo "  dashboard   : outputs/dp_skillset/${TARGET_DATASET}/${DP_EVAL_OUTPUT_SUFFIX}/index.html"
echo "  mode        : direct (no sbatch/srun, no GPU allocation)"
FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
  "${EVAL_SRC_DIR}/eval.sbatch"
