#!/usr/bin/env bash
# Inputs:
#   config : ./dp_eval_config.yaml  (roots + DP selection + HTML knobs + slurm)
# Outputs:
#   DP : ./outputs/dp_skillset/{dataset}/{dp_tag}_ck{ckpt}{suffix}.html
#
# Submit the DP skill-boundary eval (boxed start/end frames per skill + the
# multimodality curve). FSQ-independent; use submit_fsq_eval.sh for the FSQ eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SRC_DIR="${SCRIPT_DIR}/src"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${TRAIN_SKILLS_CONFIG:-${SCRIPT_DIR}/dp_eval_config.yaml}}"

# This script runs exactly the DP eval; the shared eval.sbatch honours these (env wins over yaml).
export EVAL_RUN_DP=true
export EVAL_RUN_FSQ=false

# Freeze the config so this job ignores later edits to the repo yaml (see configs/snapshot_config.sh).
_lib="$(dirname "${EVAL_CONFIG}")"; while [ ! -f "${_lib}/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/snapshot_config.sh"
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

# The selected artifact must already exist. Its manifest replaces all duplicated
# DP/checkpoint/dataset/threshold fields that used to live in this eval yaml.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
if [[ "${DP_EVAL_SKILLSET_DIR}" = /* ]]; then
  SELECTED_SKILLSET_DIR="${DP_EVAL_SKILLSET_DIR}"
else
  SELECTED_SKILLSET_DIR="${PROJECT_ROOT}/${DP_EVAL_SKILLSET_DIR}"
fi
DP_MANIFEST="${SELECTED_SKILLSET_DIR}/skillset_manifest.json"
if [ ! -f "${DP_MANIFEST}" ]; then
  echo "Skillset manifest not found: ${DP_MANIFEST}" >&2
  exit 1
fi
if ! TARGET_DATASET="$("${BOOTSTRAP_PYTHON}" - "${DP_MANIFEST}" "${SELECTED_SKILLSET_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
skillset_dir = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
dataset_name = str(manifest.get("dataset_name", "")).strip()
dataset_dir_raw = str(manifest.get("dataset_dir", "")).strip()
policy_raw = str(manifest.get("policy_path", "")).strip()
if not dataset_name:
    raise ValueError("manifest dataset_name is empty")
if not dataset_dir_raw:
    raise ValueError("manifest dataset_dir is empty")
dataset_dir = Path(dataset_dir_raw)
if not policy_raw or len(Path(policy_raw).parents) < 3:
    raise ValueError(f"invalid manifest policy_path: {policy_raw!r}")
if not (skillset_dir / "skills").is_dir():
    raise FileNotFoundError(f"skillset skills directory not found: {skillset_dir / 'skills'}")
if not (dataset_dir / "videos").is_dir():
    raise FileNotFoundError(f"source dataset videos not found: {dataset_dir / 'videos'}")
print(dataset_name)
PY
  )"; then
  echo "Invalid skillset manifest: ${DP_MANIFEST}" >&2
  exit 1
fi

SBATCH_ARGS=(
  --job-name=dp_eval
  --partition="${FSQ_EVAL_PARTITION}"
  --qos="${FSQ_EVAL_QOS}"
  --gres="${FSQ_EVAL_GRES}"
  --cpus-per-task="${FSQ_EVAL_CPUS_PER_TASK}"
  --mem="${FSQ_EVAL_MEM}"
  --time="${FSQ_EVAL_TIME}"
)
if [ -n "${FSQ_EVAL_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${FSQ_EVAL_NODELIST}")
fi
if [ -n "${FSQ_EVAL_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${FSQ_EVAL_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs outputs

echo "Submit DP skill-boundary eval"
echo "  skillset    : ${SELECTED_SKILLSET_DIR}"
echo "  dataset     : ${TARGET_DATASET} (from manifest)"
echo "  slurm       : partition=${FSQ_EVAL_PARTITION} qos=${FSQ_EVAL_QOS} gres=${FSQ_EVAL_GRES}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # Inside an existing allocation (e.g. salloc) → reuse the held GPU as a job
  # step instead of queueing a fresh job. Resources come from the allocation,
  # so SBATCH_ARGS are ignored here; the config snapshot still applies.
  echo "  mode        : srun (reusing allocation ${SLURM_JOB_ID})"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
    srun "${EVAL_SRC_DIR}/eval.sbatch"
else
  echo "  mode        : sbatch (new job)"
  FSQ_EVAL_DIR="${SCRIPT_DIR}" FSQ_EVAL_CONFIG="${EVAL_CONFIG}" \
    sbatch "${SBATCH_ARGS[@]}" "${EVAL_SRC_DIR}/eval.sbatch"
fi
