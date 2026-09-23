#!/usr/bin/env bash
# Inputs:
#   config     : ../dp_eval_config.yaml or ../fsq_eval_config.yaml, passed via
#                FSQ_EVAL_CONFIG by the submit scripts
#   toggles    : EVAL_RUN_DP / EVAL_RUN_FSQ env, forced by submit_dp_eval.sh /
#                submit_fsq_eval.sh (env wins over yaml)
#   FSQ model  : resolved from {outputs_root}/FSQ/{fsq_eval_run_name}
# Outputs (per eval_run_fsq / eval_run_dp toggles):
#   FSQ : {skill_eval}/outputs/{run}/{epoch}/fsq_eval.html  (+ skill_latents.npz next to ckpt)
#   DP  : {skill_eval}/outputs/dp_skillset/{dataset}/{output_suffix}/index.html
#
# Shared eval job body — runs the FSQ reconstruction eval and/or the DP skill-boundary eval.
#SBATCH --job-name=skill_eval
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Anchor on FSQ_EVAL_DIR passed explicitly by the submit script — correct for both sbatch and
# srun-in-salloc (where SLURM_SUBMIT_DIR points to the allocation's launch dir, not FSQ_eval).
# Fall back to SLURM_SUBMIT_DIR, then to the spool path.
EVAL_DIR="${FSQ_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # skill_eval
# Two configs exist now (dp_eval_config.yaml / fsq_eval_config.yaml), so there is no safe
# default — always launch through submit_dp_eval.sh / submit_fsq_eval.sh, which pass this.
EVAL_CONFIG="${FSQ_EVAL_CONFIG:?launch via submit_dp_eval.sh or submit_fsq_eval.sh}"

BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# An FSQ array element selects exactly one checkpoint from the frozen fanout.
CHECKPOINT_OVERRIDE="${FSQ_EVAL_CHECKPOINT_OVERRIDE:-}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${FSQ_EVAL_FANOUT:-}" ]; then
  read -r -a FSQ_CHECKPOINTS <<< "${FSQ_EVAL_FANOUT}"
  if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#FSQ_CHECKPOINTS[@]}" ]; then
    echo "FSQ array index ${SLURM_ARRAY_TASK_ID} is out of range." >&2
    exit 1
  fi
  CHECKPOINT_OVERRIDE="${FSQ_CHECKPOINTS[${SLURM_ARRAY_TASK_ID}]}"
fi

# Eval knobs and the selected FSQ artifact are resolved from fsq_meta.json.
EVAL_CONFIG_ARGS=(--config "${EVAL_CONFIG}")
if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
  EVAL_CONFIG_ARGS+=(--checkpoint "${CHECKPOINT_OVERRIDE}")
fi
if ! RESOLVED_EVAL="$(
  "${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/eval_config.py" \
    "${EVAL_CONFIG_ARGS[@]}" --shell
)"; then
  echo "FSQ evaluation bootstrap failed for checkpoint ${CHECKPOINT_OVERRIDE:-default}." >&2
  exit 1
fi
eval "${RESOLVED_EVAL}"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START DP_FSQ EVAL"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "run_fsq : ${EVAL_RUN_FSQ}    run_dp : ${EVAL_RUN_DP}"

# DP boundary visualization is a CPU-only reader/renderer. Keep login-node CPU
# usage bounded. FSQ reconstruction remains a GPU workload and retains its
# CUDA guard when launched through submit_fsq_eval.sh.
if [ "${EVAL_RUN_FSQ}" = "true" ]; then
  nvidia-smi || true
  source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/gpu_guard.sh"
  require_cuda_or_requeue "${PROJECT_ROOT}/.venv/bin/python"
else
  DP_CPU_THREADS="${DP_EVAL_CPU_THREADS:-4}"
  export OMP_NUM_THREADS="${DP_CPU_THREADS}"
  export MKL_NUM_THREADS="${DP_CPU_THREADS}"
  export OPENBLAS_NUM_THREADS="${DP_CPU_THREADS}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-skill_eval}_matplotlib}"
  mkdir -p "${MPLCONFIGDIR}"
  echo "GPU     : not required (CPU-only DP rendering)"
fi

if [ "${EVAL_RUN_FSQ}" != "true" ] && [ "${EVAL_RUN_DP}" != "true" ]; then
  echo "Both eval_run_fsq and eval_run_dp are false — nothing to do." >&2
  exit 1
fi

# ── DP skill-boundary eval (FSQ-independent; off the FSQ_dataset skillset npz + curves) ──
if [ "${EVAL_RUN_DP}" = "true" ]; then
  DP_TASK_ARGS=()
  [ -n "${DP_EVAL_TASK_IDS}" ] && DP_TASK_ARGS=(--task-ids ${DP_EVAL_TASK_IDS})
  DP_VIDEO_ARGS=()
  [ "${DP_EVAL_SKILL_VIDEO}" = "true" ] && DP_VIDEO_ARGS=(--skill-video)
  DP_DISPLAY_ARGS=()
  [ "${DP_EVAL_SHOW_START_END_FRAMES}" != "true" ] && DP_DISPLAY_ARGS+=(--hide-start-end-frames)
  [ "${DP_EVAL_SHOW_COS_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide-cos-graph)
  [ "${DP_EVAL_SHOW_GAIN_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide-gain-graph)
  [ "${DP_EVAL_SHOW_BIC_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide-bic-graph)
  [ "${DP_EVAL_SHOW_GRIPPER_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide-gripper-graph)
  echo ""
  echo "DP skill-boundary comparison → ${EVAL_DIR}/outputs/dp_skillset"
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u "${EVAL_DIR}/src/run_dp_skillset_comparison.py" \
      --skillsets-json "${DP_EVAL_SKILLSETS_JSON}" \
      --out-root "${EVAL_DIR}/outputs/dp_skillset" \
      --output-suffix "${DP_EVAL_OUTPUT_SUFFIX:-comparison}" \
      --n-episodes "${DP_EVAL_N_EPISODES}" \
      --task-id-space "${DP_EVAL_TASK_ID_SPACE}" \
      --target-task "${DP_EVAL_TARGET_TASK}" \
      --thumb-size "${FSQ_EVAL_THUMB_SIZE}" \
      ${DP_VIDEO_ARGS[@]+"${DP_VIDEO_ARGS[@]}"} \
      ${DP_DISPLAY_ARGS[@]+"${DP_DISPLAY_ARGS[@]}"} \
      ${DP_TASK_ARGS[@]+"${DP_TASK_ARGS[@]}"}
  echo "DONE DP comparison"
fi

# ── FSQ reconstruction eval (encoder + decoder → wandb scalars + interactive HTML) ──
if [ "${EVAL_RUN_FSQ}" = "true" ]; then
  MODEL_PATH="${FSQ_EVAL_MODEL_PATH}"
  EPOCH_TAG="${FSQ_EVAL_EPOCH_TAG}"
  LATENTS_PATH="${FSQ_EVAL_LATENTS_PATH}"
  F_SKILLSET="${FSQ_EVAL_SKILLSET_DIR}"
  F_DATASET_VIDEO_DIR="${FSQ_EVAL_DATASET_DIR}"
  # HTML stays under skill_eval/outputs; latents live next to the checkpoint in outputs/FSQ.
  OUTPUT_DIR="${EVAL_DIR}/outputs/${FSQ_EVAL_RUN_NAME}/${EPOCH_TAG}"
  echo ""
  echo "FSQ eval → ${OUTPUT_DIR}  (model ${MODEL_PATH})"
  echo "  source skillset: ${F_SKILLSET}"
  echo "  source dataset : ${F_DATASET_VIDEO_DIR}"
  FSQ_RANDOM_SKILL_ARGS=()
  if [ "${FSQ_EVAL_RANDOM_FAR_SKILL}" = "true" ]; then
    FSQ_RANDOM_SKILL_ARGS=(
      --random_far_skill
      --random_far_fraction "${FSQ_EVAL_RANDOM_FAR_FRACTION}"
    )
  fi
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/fsq_eval.py \
      --model_path          "${MODEL_PATH}" \
      --skills_dir          "${F_SKILLSET}/skills" \
      --dataset_dir      "${F_DATASET_VIDEO_DIR}" \
      --output_dir       "${OUTPUT_DIR}" \
      --latents_path     "${LATENTS_PATH}" \
      --n_action_steps   "${FSQ_EVAL_N_ACTION_STEPS}" \
      --max_plot_samples "${FSQ_EVAL_N_SAMPLES}" \
      --max_plot_entries "${FSQ_EVAL_MAX_ENTRIES}" \
      --decoder_scope    "${FSQ_EVAL_DECODER_SCOPE}" \
      --thumb_size       "${FSQ_EVAL_THUMB_SIZE}" \
      --end_threshold    "${FSQ_EVAL_END_THRESHOLD}" \
      --wandb_project    "${FSQ_EVAL_WANDB_PROJECT}" \
      --wandb_run_name   "${FSQ_EVAL_RUN_NAME}_${EPOCH_TAG}" \
      ${FSQ_RANDOM_SKILL_ARGS[@]+"${FSQ_RANDOM_SKILL_ARGS[@]}"}
  echo "DONE FSQ → ${OUTPUT_DIR}/fsq_eval.html"
fi
