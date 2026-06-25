#!/usr/bin/env bash
# SkillVLA FT closed-loop EVAL on LIBERO sim via lerobot_skillvla_eval.
# Rolls out a FINETUNED policy; the terminator is the FT-adapted FSQ_ft.pt (overrides the checkpoint's
# fsq_path). train_terminator is forced false so no co-training terminator is built at eval.
#SBATCH --job-name=ft_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${FT_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # FT_eval
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FT_EVAL_CONFIG:-${SUBMIT_DIR}/ft_eval_config.yaml}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_eval_config.py" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH          # avoid system cuDNN shadowing the bundled one

LIBERO_REPO_ROOT="${PROJECT_ROOT}/tools/lerobot-libero"
LIBERO_SOURCE_ROOT="${LIBERO_REPO_ROOT}/libero"
LIBERO_BENCHMARK_ROOT="${LIBERO_SOURCE_ROOT}/libero"
LIBERO_LOCAL_CONFIG_DIR="${LIBERO_REPO_ROOT}/.libero_config"
if [ ! -d "${LIBERO_BENCHMARK_ROOT}/init_files" ]; then
  echo "LIBERO init_files not found: ${LIBERO_BENCHMARK_ROOT}/init_files" >&2
  exit 1
fi
mkdir -p "${LIBERO_LOCAL_CONFIG_DIR}"
cat > "${LIBERO_LOCAL_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_SOURCE_ROOT}/datasets
assets: ${LIBERO_BENCHMARK_ROOT}/assets
EOF
export LIBERO_CONFIG_PATH="${LIBERO_LOCAL_CONFIG_DIR}"

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}  (train FT first)" >&2
  exit 1
fi

# Terminator to gate skill transitions (action chunk always from the expert). EVAL_TERMINATOR:
#   ft     → FT-adapted terminator co-trained INTO this checkpoint (checkpoints/<ckpt>/FSQ_ft.pt);
#            lazy-exported from the checkpoint's model.safetensors if missing, else falls back to base.
#   base   → the new task's ORIGINAL dataset FSQ.pt (terminator before any in-policy co-training).
#   stage2 → the terminator FT inherited from its Stage-2 source checkpoint (<stage2_ckpt>/FSQ_ft.pt;
#            exported by Stage-2 training; lazy-exported from the Stage-2 checkpoint here if missing).
EXPORT_TERM="${PROJECT_ROOT}/.venv/bin/python ${SRC_DIR}/../../FT/src/export_ft_terminator.py"
case "${EVAL_TERMINATOR}" in
  base)
    FSQ_FOR_EVAL="${BASE_FSQ}"
    ;;
  stage2)
    if [ ! -f "${STAGE2_FSQ_PATH}" ]; then
      echo "Stage-2 terminator missing → exporting from ${STAGE2_RUN_DIR} @ ${STAGE2_CHECKPOINT}"
      ${EXPORT_TERM} --ft_run_dir="${STAGE2_RUN_DIR}" --checkpoint="${STAGE2_CHECKPOINT}" \
        --base_fsq="${BASE_FSQ}" --out="${STAGE2_FSQ_PATH}" --skip_if_no_terminator
    fi
    if [ ! -f "${STAGE2_FSQ_PATH}" ]; then
      echo "eval_terminator=stage2 but no Stage-2 terminator found/exportable: ${STAGE2_FSQ_PATH}" >&2
      echo "  (was the Stage-2 run trained with train_terminator=true?)" >&2
      exit 1
    fi
    FSQ_FOR_EVAL="${STAGE2_FSQ_PATH}"
    ;;
  *)  # ft (default)
    FSQ_FOR_EVAL="${BASE_FSQ}"
    if [ "${TRAIN_TERMINATOR_USED}" = "true" ]; then
      if [ ! -f "${FT_FSQ_PATH}" ]; then
        echo "Adapted terminator missing for ${CHECKPOINT} → exporting from the checkpoint"
        ${EXPORT_TERM} --ft_run_dir="${FT_RUN_DIR}" --checkpoint="${CHECKPOINT}" \
          --base_fsq="${BASE_FSQ}" --out="${FT_FSQ_PATH}" --skip_if_no_terminator
      fi
      [ -f "${FT_FSQ_PATH}" ] && FSQ_FOR_EVAL="${FT_FSQ_PATH}"
    fi
    ;;
esac

echo "START SkillVLA FT EVAL"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "policy   : ${POLICY_PATH}"
echo "fsq(term): ${FSQ_FOR_EVAL}"
echo "target   : ${TARGET_TASK}  task_ids=${TASK_IDS}  episodes=${N_EPISODES}/task"
nvidia-smi || true

if [ ! -f "${FSQ_FOR_EVAL}" ]; then
  echo "No terminator FSQ available (neither adapted nor base): ${FSQ_FOR_EVAL}" >&2
  exit 1
fi

# skill_html needs skill_latents.npz; build_data may have cleaned it up → auto-disable.
SKILL_HTML_EFFECTIVE="${SKILL_HTML}"
if [ "${SKILL_HTML}" = "true" ] && [ ! -f "${SKILL_LATENTS_PATH}" ]; then
  echo "[warn] skill_latents.npz missing (${SKILL_LATENTS_PATH}); disabling skill_html."
  SKILL_HTML_EFFECTIVE=false
fi

mkdir -p "${EVAL_OUT_DIR}"

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" \
  "${PROJECT_ROOT}/.venv/bin/python" -m lerobot.scripts.lerobot_skillvla_eval \
    --policy.path="${POLICY_PATH}" \
    --policy.fsq_path="${FSQ_FOR_EVAL}" \
    --policy.train_terminator=false \
    --env.type=libero \
    --env.task="${TARGET_TASK}" \
    --env.fps=20 \
    --env.task_ids="${TASK_IDS}" \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.max_videos_per_task="${MAX_VIDEOS_PER_TASK}" \
    --eval.video_frame_stride="${VIDEO_FRAME_STRIDE}" \
    --eval.video_fps="${VIDEO_FPS}" \
    --eval.skill_html="${SKILL_HTML_EFFECTIVE}" \
    --eval.skill_html_train_samples="${SKILL_HTML_TRAIN_SAMPLES}" \
    --eval.skill_html_skill_latents_path="${SKILL_LATENTS_PATH}" \
    --eval.skill_html_raw_dataset_dir="${RAW_DATASET_DIR}" \
    --eval.skill_html_image_key="${IMAGE_KEY}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.device=cuda \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --policy.skill_end_mode="${SKILL_END_MODE}" \
    --policy.skill_end_threshold="${SKILL_END_THRESHOLD}" \
    --policy.skill_end_progress_threshold="${SKILL_END_PROGRESS_THRESHOLD}" \
    --policy.inference_skill_max_length="${INFERENCE_SKILL_MAX_LENGTH}" \
    --policy.use_gt_skill="${USE_GT_SKILL}" \
    --policy.gt_skill_dataset_dir="${GT_SKILL_DATASET_DIR}" \
    --policy.skill_advance_mode="${SKILL_ADVANCE_MODE}" \
    --wandb_project="${WANDB_PROJECT}" \
    --job_name="${WANDB_RUN_NAME}" \
    --output_dir="${EVAL_OUT_DIR}" \
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
    ${EXTRA_ARGS:-}

echo "DONE → ${EVAL_OUT_DIR}"
