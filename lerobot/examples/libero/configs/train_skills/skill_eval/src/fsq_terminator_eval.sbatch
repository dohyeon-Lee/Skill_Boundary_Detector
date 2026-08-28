#!/usr/bin/env bash
# Score one FSQ co-trained terminator against the GT skill boundaries.
# One array task per terminator_models entry.
#SBATCH --job-name=FSQ_TERM
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

SUBMIT_DIR="${FSQ_TERM_EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FSQ_TERM_EVAL_CONFIG:-${SUBMIT_DIR}/fsq_terminator_eval_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

: "${FSQ_TERM_MODEL_LABELS:?fsq_terminator_eval needs FSQ_TERM_MODEL_LABELS}"
read -r -a LABELS <<< "${FSQ_TERM_MODEL_LABELS}"
INDEX="${FSQ_TERM_ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if [ "${INDEX}" -ge "${#LABELS[@]}" ]; then
  echo "Invalid array index ${INDEX} for ${#LABELS[@]} model(s)." >&2
  exit 1
fi
MODEL_LABEL="${LABELS[${INDEX}]}"

SETTINGS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/fsq_terminator_eval_config.py" \
    --config "${CONFIG_PATH}" --model-label "${MODEL_LABEL}" --shell
)"
eval "${SETTINGS}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

for artifact in "${FSQ_MODEL_PATH}" "${FSQ_SKILLS_DIR}" "${SKILL_DATASET_DIR}"; do
  if [ ! -e "${artifact}" ]; then
    echo "Missing FSQ terminator eval artifact: ${artifact}" >&2
    exit 1
  fi
done
mkdir -p "${SUBMIT_DIR}/logs" "${EVAL_OUT_DIR}"

echo "START FSQ TERMINATOR PROBE"
echo "node      : ${SLURMD_NODENAME:-${HOSTNAME:-unknown}}"
echo "model     : ${FSQ_MODEL_LABEL} (${FSQ_MODEL_RUN} ${FSQ_MODEL_EPOCH_TAG})"
echo "suite     : ${TARGET_TASK} task_ids=${TASK_IDS}"
echo "episodes  : ${EPISODES_PER_TASK}/task selection=${EPISODE_SELECTION}"
echo "threshold : ${EVAL_END_THRESHOLD}"
echo "output    : ${EVAL_OUT_DIR}"
nvidia-smi || true

# Image-conditioned terminators run a vision backbone per frame, so a node whose
# CUDA is broken must requeue rather than crawl on CPU. Use the INLINE guard:
# require_cuda_or_requeue would spawn a second interpreter just to import torch,
# which costs minutes on this cluster's shared venv (measured 9 min of start-up
# per task, most of it two torch imports).
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
prepare_inline_cuda_guard

RESUME_ARGS=()
[ "${EVAL_RESUME}" != "true" ] || RESUME_ARGS=(--resume)

# Loading the skillset and the first model takes minutes; a node that hangs
# before the runner's first log line is stuck, not slow.
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/stall_guard.sh"
PYTHONPATH="${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/examples/libero" \
  run_with_startup_guard "${STALL_GUARD_TIMEOUT:-900}" "terminator variant" -- \
  "${PROJECT_ROOT}/.venv/bin/python" "${SRC_DIR}/run_fsq_terminator_eval.py" \
    --model-path "${FSQ_MODEL_PATH}" \
    --terminator-overlay "${FSQ_TERMINATOR_OVERLAY_PATH}" \
    --model-source "${FSQ_MODEL_SOURCE}" \
    --code-space-id "${FSQ_CODE_SPACE_ID}" \
    --model-label "${FSQ_MODEL_LABEL}" \
    --model-run "${FSQ_MODEL_RUN}" \
    --epoch-tag "${FSQ_MODEL_EPOCH_TAG}" \
    --skills-dir "${FSQ_SKILLS_DIR}" \
    --dataset-dir "${SKILL_DATASET_DIR}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --task-ids "${TASK_IDS}" \
    --task-id-space "${TASK_ID_SPACE}" \
    --target-task "${TARGET_TASK}" \
    --episode-ids "${EPISODE_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --episode-selection "${EPISODE_SELECTION}" \
    --end-threshold "${EVAL_END_THRESHOLD}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --seed "${EVAL_SEED}" \
    --collection-dir "${EVAL_COLLECTION_DIR}" \
    --expected-labels "${FSQ_MODEL_LABELS}" \
    --max-entries "${EVAL_MAX_PLOT_ENTRIES}" \
    --max-samples "${EVAL_MAX_PLOT_SAMPLES}" \
    --fps "${VIDEO_FPS}" \
    --frame-stride "${VIDEO_FRAME_STRIDE}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} || RUN_STATUS=$?
RUN_STATUS="${RUN_STATUS:-0}"
# Requeues onto another node when the marker says CUDA was missing; otherwise a
# failed run must stop here rather than falling through to the report step.
handle_inline_cuda_guard_exit "${RUN_STATUS}"
[ "${RUN_STATUS}" -eq 0 ] || exit "${RUN_STATUS}"

echo "DONE -> ${EVAL_COLLECTION_DIR}/index.html"
