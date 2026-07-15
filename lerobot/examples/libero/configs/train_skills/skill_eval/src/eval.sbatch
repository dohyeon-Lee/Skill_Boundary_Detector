#!/usr/bin/env bash
# Inputs:
#   roots      : ../../train_skills_config.yaml
#   eval knobs : ../eval_config.yaml
#   FSQ model  : {project_root}/outputs/FSQ/{fsq_eval_run_name}/FSQ.pt (or FSQ_epoch{N:04d}.pt)
# Outputs (per eval_run_fsq / eval_run_dp toggles):
#   FSQ : {skill_eval}/outputs/{run}/{epoch}/fsq_eval.html  (+ skill_latents.npz next to ckpt)
#   DP  : {skill_eval}/outputs/dp_skillset/{dataset}/index.html
#
# DP_FSQ eval — FSQ reconstruction eval and/or the DP skill-boundary eval (toggled in yaml).
#SBATCH --job-name=skill_eval
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
EVAL_DIR="${FSQ_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # skill_eval
COMMON_SRC_DIR="${EVAL_DIR}/../src"                          # train_skills/src
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-${EVAL_DIR}/eval_config.yaml}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:-${EVAL_DIR}/eval_config.yaml}"

BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# eval knobs FIRST so DP-selection overrides can steer the shared resolver below.
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/eval_config.py" --config "${EVAL_CONFIG}" --shell)"
[ -n "${EVAL_DP_RUN_NAME}" ]   && export DP_RUN_NAME="${EVAL_DP_RUN_NAME}"
[ -n "${EVAL_DP_CHECKPOINT}" ] && export DP_CHECKPOINT="${EVAL_DP_CHECKPOINT}"
if [ -n "${TRAIN_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TRAIN_DATA}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
fi

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
  DP_SKILLSET_DIR="${SKILLSET_DIR}"
  DP_DATASET_VIDEO_DIR="${DATASET_VIDEO_DIR}"
  DP_IMAGE_KEY="${FSQ_EVAL_IMAGE_KEY}"
  DP_PROBE_SUFFIX="${SKILLSET_PROBE_SUFFIX}"
  if [ -n "${DP_EVAL_SKILLSET_DIR:-}" ]; then
    if [[ "${DP_EVAL_SKILLSET_DIR}" = /* ]]; then
      DP_SKILLSET_DIR="${DP_EVAL_SKILLSET_DIR}"
    else
      DP_SKILLSET_DIR="${PROJECT_ROOT}/${DP_EVAL_SKILLSET_DIR}"
    fi
    DP_MANIFEST="${DP_SKILLSET_DIR}/skillset_manifest.json"
    if [ ! -f "${DP_MANIFEST}" ]; then
      echo "Skillset manifest not found: ${DP_MANIFEST}" >&2
      exit 1
    fi
    DP_DATASET_VIDEO_DIR=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_dir"])' "${DP_MANIFEST}")
    DP_IMAGE_KEY=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("image_key", "observation.images.image"))' "${DP_MANIFEST}")
    DP_MODE=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode", "custom"))' "${DP_MANIFEST}")
    DP_PROBE_SUFFIX="_${DP_MODE}"
  fi
  # Shallow folder + DP-named HTML so different DPs don't overwrite each other (no deep nesting).
  # Strip the redundant dataset prefix from the run name (libero_90_full_full_state_obs10 → state_obs10).
  DP_TAG="${DP_POLICY#${TARGET_DATASET}_}"
  DP_OUT_DIR="${EVAL_DIR}/outputs/dp_skillset/${TARGET_DATASET}"
  # Probe suffix is empty for legacy spherical_xyz and tagged for pca_action, so
  # side-by-side evaluations of the same DP/checkpoint never overwrite each other.
  DP_OUT_HTML="${DP_TAG}_ck${DP_CHECKPOINT}${DP_PROBE_SUFFIX}.html"
  DP_TASK_ARGS=()
  [ -n "${DP_EVAL_TASK_IDS}" ] && DP_TASK_ARGS=(--task_ids ${DP_EVAL_TASK_IDS})
  echo ""
  echo "DP skill-boundary eval → ${DP_OUT_DIR}"
  if [ ! -d "${DP_SKILLSET_DIR}/curves" ]; then
    echo "  [note] no ${DP_SKILLSET_DIR}/curves — rendering frames only (no multimodality graph)."
    echo "         Rebuild the skillset (build_skill_dataset now writes curves) to get the graph."
  fi
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/dp_skillset_eval.py \
      --skills_dir  "${DP_SKILLSET_DIR}/skills" \
      --curves_dir  "${DP_SKILLSET_DIR}/curves" \
      --dataset_dir "${DP_DATASET_VIDEO_DIR}" \
      --image_key   "${DP_IMAGE_KEY}" \
      --out_dir     "${DP_OUT_DIR}" \
      --out_html    "${DP_OUT_HTML}" \
      --n_episodes  "${DP_EVAL_N_EPISODES}" \
      --thumb_size  "${FSQ_EVAL_THUMB_SIZE}" \
      ${DP_TASK_ARGS[@]+"${DP_TASK_ARGS[@]}"}
  echo "DONE DP → ${DP_OUT_DIR}/${DP_OUT_HTML}"
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
  # HTML stays under skill_eval/outputs; latents live next to the checkpoint in outputs/FSQ.
  OUTPUT_DIR="${EVAL_DIR}/outputs/${FSQ_EVAL_RUN_NAME}/${EPOCH_TAG}"
  echo ""
  echo "FSQ eval → ${OUTPUT_DIR}  (model ${MODEL_PATH})"
  if [ ! -f "${MODEL_PATH}" ]; then
    echo "FSQ model not found: ${MODEL_PATH}" >&2
    echo "Check fsq_eval_run_name / checkpoint in ${EVAL_CONFIG}" >&2
    exit 1
  fi
  # Evaluate THIS FSQ on the skillset/cameras it was TRAINED on — read from fsq_meta.json (written by
  # train_fsq). Independent of eval_dp_run_name (that only steers the DP eval). The FSQ's skillset/
  # tokens already exist (training required them). Fall back to eval_config if no meta (older runs).
  FSQ_META="${MODEL_DIR}/fsq_meta.json"
  F_SKILLSET="${SKILLSET_DIR}"
  F_USE_THIRD="${FSQ_TERMINATOR_USE_THIRD}"; F_USE_WRIST="${FSQ_TERMINATOR_USE_WRIST}"
  if [ -f "${FSQ_META}" ]; then
    read -r F_DP F_CKPT F_GRID F_USE_THIRD F_USE_WRIST < <("${BOOTSTRAP_PYTHON}" - "${FSQ_META}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["dp_run_name"], m["dp_checkpoint"], m["fsq_patch_grid"],
      str(m["fsq_terminator_use_third"]).lower(), str(m["fsq_terminator_use_wrist"]).lower())
PY
)
    # Re-resolve the FSQ's own skillset for its recorded DP/grid (subshell, no leak).
    eval "$(DP_RUN_NAME="${F_DP}" DP_CHECKPOINT="${F_CKPT}" FSQ_PATCH_GRID="${F_GRID}" \
      "${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell 2>/dev/null \
      | grep -E '^export SKILLSET_DIR=')"
    F_SKILLSET="${SKILLSET_DIR}"
    echo "  FSQ source DP    : ${F_DP} (ck${F_CKPT}, pg${F_GRID}, third=${F_USE_THIRD} wrist=${F_USE_WRIST})"
  fi
  if [ ! -d "${F_SKILLSET}/skills" ]; then
    echo "FSQ's training skillset not found: ${F_SKILLSET}/skills" >&2; exit 1
  fi
  # ONLINE: fsq_eval이 --dataset_dir(raw mp4)에서 DINO를 직접 warm-pass 인코딩 — per-frame DINO dir
  # 불필요 (dino-prep 빌드 단계 제거; train_FSQ와 동일 계약이라 학습·평가가 같은 토큰을 봄).
  # Camera→slot 규약은 기존 그대로: 3rd → primary slot (--image_key), wrist → wrist slot
  # (--image_key_wrist, terminator가 wrist를 쓸 때 항상 — wrist-only는 primary도 wrist).
  if [ "${F_USE_THIRD}" = "true" ]; then FSQ_PRIMARY_KEY="observation.images.image"; else FSQ_PRIMARY_KEY="observation.images.wrist_image"; fi
  FSQ_DINO_ARGS=(--image_key "${FSQ_PRIMARY_KEY}")
  if [ "${F_USE_WRIST}" = "true" ]; then
    FSQ_DINO_ARGS+=(--image_key_wrist "observation.images.wrist_image")
  fi
  PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
    python -u examples/libero/fsq_eval.py \
      --model_path          "${MODEL_PATH}" \
      --skills_dir          "${F_SKILLSET}/skills" \
      "${FSQ_DINO_ARGS[@]}" \
      --dataset_dir      "${DATASET_VIDEO_DIR}" \
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
