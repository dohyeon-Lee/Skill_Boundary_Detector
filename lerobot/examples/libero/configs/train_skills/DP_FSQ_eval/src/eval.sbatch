#!/usr/bin/env bash
# Inputs:
#   roots      : ../../train_skills_config.yaml
#   eval knobs : ../eval_config.yaml
#   FSQ model  : {project_root}/outputs/FSQ/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch{N:04d}.pt)
# Outputs (per eval_run_fsq / eval_run_dp toggles):
#   FSQ : {DP_FSQ_eval}/outputs/{run}/{epoch}/fsq_eval.html  (+ skill_latents.npz next to ckpt)
#   DP  : {DP_FSQ_eval}/outputs/dp_skillset/{dataset}/index.html
#
# DP_FSQ eval — FSQ reconstruction eval and/or the DP skill-boundary eval (toggled in yaml).
#SBATCH --job-name=DP_FSQ_eval
#SBATCH --gres=gpu:1
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
EVAL_DIR="${FSQ_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # DP_FSQ_eval
COMMON_SRC_DIR="${EVAL_DIR}/../src"                          # train_skills/src
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${EVAL_DIR}/../train_skills_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${EVAL_DIR}/eval_config.yaml}"

BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/eval_config.py" --config "${EVAL_CONFIG}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=""
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${PROJECT_ROOT}/../.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${PROJECT_ROOT}/../.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

DATASET_VIDEO_DIR="${DATASET_ROOT}/${TARGET_DATASET}"

echo "START DP_FSQ EVAL"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "run_fsq : ${EVAL_RUN_FSQ}    run_dp : ${EVAL_RUN_DP}"
echo "Skills  : ${SKILLSET_DIR}/skills"
echo "Dataset : ${DATASET_VIDEO_DIR}"
nvidia-smi || true

if [ "${EVAL_RUN_FSQ}" != "true" ] && [ "${EVAL_RUN_DP}" != "true" ]; then
  echo "Both eval_run_fsq and eval_run_dp are false — nothing to do." >&2
  exit 1
fi

# ── DP skill-boundary eval (FSQ-independent; off the FSQ_dataset skillset npz + curves) ──
if [ "${EVAL_RUN_DP}" = "true" ]; then
  DP_OUT_DIR="${EVAL_DIR}/outputs/dp_skillset/${TARGET_DATASET}"
  DP_TASK_ARGS=()
  [ -n "${DP_EVAL_TASK_IDS}" ] && DP_TASK_ARGS=(--task_ids ${DP_EVAL_TASK_IDS})
  echo ""
  echo "DP skill-boundary eval → ${DP_OUT_DIR}"
  if [ ! -d "${SKILLSET_DIR}/curves" ]; then
    echo "  [note] no ${SKILLSET_DIR}/curves — rendering frames only (no multimodality graph)."
    echo "         Rebuild the skillset (build_skill_dataset now writes curves) to get the graph."
  fi
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/dp_skillset_eval.py \
      --skills_dir  "${SKILLSET_DIR}/skills" \
      --curves_dir  "${SKILLSET_DIR}/curves" \
      --dataset_dir "${DATASET_VIDEO_DIR}" \
      --image_key   "${FSQ_EVAL_IMAGE_KEY}" \
      --out_dir     "${DP_OUT_DIR}" \
      --n_episodes  "${DP_EVAL_N_EPISODES}" \
      --thumb_size  "${FSQ_EVAL_THUMB_SIZE}" \
      ${DP_TASK_ARGS[@]+"${DP_TASK_ARGS[@]}"}
  echo "DONE DP → ${DP_OUT_DIR}/index.html"
fi

# ── FSQ reconstruction eval (encoder + decoder → wandb scalars + interactive HTML) ──
if [ "${EVAL_RUN_FSQ}" = "true" ]; then
  FSQ_OUTPUTS_ROOT="$(dirname "${FSQ_OUTPUT_DIR}")"
  MODEL_DIR="${FSQ_OUTPUTS_ROOT}/${FSQ_EVAL_RUN_NAME}"
  if [ "${FSQ_EVAL_CHECKPOINT}" = "0" ]; then
    MODEL_PATH="${MODEL_DIR}/FSQ.pt"
    EPOCH_TAG="best"
    LATENTS_PATH="${MODEL_DIR}/skill_latents.npz"
  else
    MODEL_PATH="${MODEL_DIR}/FSQ_epoch$(printf '%04d' "${FSQ_EVAL_CHECKPOINT}").pt"
    EPOCH_TAG="epoch$(printf '%04d' "${FSQ_EVAL_CHECKPOINT}")"
    LATENTS_PATH="${MODEL_DIR}/skill_latents_${EPOCH_TAG}.npz"
  fi
  # HTML stays under DP_FSQ_eval/outputs; latents live next to the checkpoint in outputs/FSQ.
  OUTPUT_DIR="${EVAL_DIR}/outputs/${FSQ_EVAL_RUN_NAME}/${EPOCH_TAG}"
  echo ""
  echo "FSQ eval → ${OUTPUT_DIR}  (model ${MODEL_PATH})"
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "FSQ model not found: ${MODEL_PATH}" >&2
    echo "Check fsq_eval_run_name / checkpoint in ${EVAL_CONFIG}" >&2
    exit 1
  fi
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/fsq_eval.py \
      --model_path          "${MODEL_PATH}" \
      --skills_dir          "${SKILLSET_DIR}/skills" \
      --dino_features       "${DINO_TOKENS_PATH}" \
      --dino_features_wrist "${DINO_TOKENS_WRIST_PATH}" \
      --dataset_dir      "${DATASET_VIDEO_DIR}" \
      --image_key        "${FSQ_EVAL_IMAGE_KEY}" \
      --output_dir       "${OUTPUT_DIR}" \
      --latents_path     "${LATENTS_PATH}" \
      --n_action_steps   "${FSQ_EVAL_N_ACTION_STEPS}" \
      --max_plot_samples "${FSQ_EVAL_N_SAMPLES}" \
      --max_plot_entries "${FSQ_EVAL_MAX_ENTRIES}" \
      --thumb_size       "${FSQ_EVAL_THUMB_SIZE}" \
      --end_threshold    "${FSQ_EVAL_END_THRESHOLD}" \
      --wandb_project    "${FSQ_EVAL_WANDB_PROJECT}" \
      --wandb_run_name   "${FSQ_EVAL_RUN_NAME}_${EPOCH_TAG}"
  echo "DONE FSQ → ${OUTPUT_DIR}/fsq_eval.html"
fi
