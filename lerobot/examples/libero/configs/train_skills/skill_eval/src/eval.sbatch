#!/usr/bin/env bash
# Inputs:
#   config     : ../dp_eval_config.yaml or ../fsq_eval_config.yaml, passed via
#                FSQ_EVAL_CONFIG by the submit scripts
#   toggles    : EVAL_RUN_DP / EVAL_RUN_FSQ env, forced by submit_dp_eval.sh /
#                submit_fsq_eval.sh (env wins over yaml)
#   FSQ model  : resolved from {outputs_root}/FSQ/{fsq_eval_run_name}
# Outputs (per eval_run_fsq / eval_run_dp toggles):
#   FSQ : {skill_eval}/outputs/{run}/{epoch}/fsq_eval.html  (+ skill_latents.npz next to ckpt)
#   DP  : {skill_eval}/outputs/dp_skillset/{dataset}/index.html
#
# Shared eval job body — runs the FSQ reconstruction eval and/or the DP skill-boundary eval.
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
# Two configs exist now (dp_eval_config.yaml / fsq_eval_config.yaml), so there is no safe
# default — always launch through submit_dp_eval.sh / submit_fsq_eval.sh, which pass these.
TRAIN_CONFIG="${TRAIN_SKILLS_CONFIG:-}"
EVAL_CONFIG="${FSQ_EVAL_CONFIG:?launch via submit_dp_eval.sh or submit_fsq_eval.sh}"

BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Eval knobs and FSQ artifacts are resolved directly from fsq_meta.json.
RESOLVED_EVAL="$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/eval_config.py" --config "${EVAL_CONFIG}" --shell)"
eval "${RESOLVED_EVAL}"
if [ "${EVAL_RUN_DP}" = "true" ]; then
  if [ -z "${TRAIN_CONFIG}" ]; then
    echo "DP eval requires TRAIN_SKILLS_CONFIG; launch via submit_dp_eval.sh" >&2
    exit 1
  fi
  if [ -n "${TRAIN_DATA:-}" ]; then
    RESOLVED_TRAIN="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --dataset "${TRAIN_DATA}" --shell)"
  else
    RESOLVED_TRAIN="$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${TRAIN_CONFIG}" --shell)"
  fi
  eval "${RESOLVED_TRAIN}"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START DP_FSQ EVAL"
echo "Node    : ${SLURMD_NODENAME:-local}"
echo "run_fsq : ${EVAL_RUN_FSQ}    run_dp : ${EVAL_RUN_DP}"
nvidia-smi || true

if [ "${EVAL_RUN_FSQ}" != "true" ] && [ "${EVAL_RUN_DP}" != "true" ]; then
  echo "Both eval_run_fsq and eval_run_dp are false — nothing to do." >&2
  exit 1
fi

# ── DP skill-boundary eval (FSQ-independent; off the FSQ_dataset skillset npz + curves) ──
if [ "${EVAL_RUN_DP}" = "true" ]; then
  DATASET_VIDEO_DIR="${DATASET_ROOT}/${TARGET_DATASET}"
  DP_SKILLSET_DIR="${SKILLSET_DIR}"
  DP_DATASET_VIDEO_DIR="${DATASET_VIDEO_DIR}"
  DP_IMAGE_KEY="${FSQ_EVAL_IMAGE_KEY}"
  DP_SUFFIX="${SKILLSET_PROBE_SUFFIX}"
  DP_OUTPUT_DATASET="${TARGET_DATASET}"
  DP_OUTPUT_POLICY="${DP_POLICY}"
  DP_OUTPUT_CHECKPOINT="${DP_CHECKPOINT}"
  DP_SUFFIX+="${SKILLSET_BOUNDARY_THRESHOLD_SUFFIX}"
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
    DP_THRESHOLD_MODE=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("detector", {}).get("boundary_threshold_mode", "episode_mean"))' "${DP_MANIFEST}")
    DP_THRESHOLD_SCALE=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("detector", {}).get("boundary_threshold_scale", 1.0))' "${DP_MANIFEST}")
    DP_OUTPUT_DATASET=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dataset_name", "dataset"))' "${DP_MANIFEST}")
    DP_MANIFEST_POLICY=$("${BOOTSTRAP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_path"])' "${DP_MANIFEST}")
    DP_OUTPUT_POLICY=$("${BOOTSTRAP_PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).parents[2].name)' "${DP_MANIFEST_POLICY}")
    DP_OUTPUT_CHECKPOINT=$("${BOOTSTRAP_PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).parent.name)' "${DP_MANIFEST_POLICY}")
    # Preserve the complete canonical artifact identity, including globalref
    # and any manual output suffix, directly from the selected seg_* folder.
    DP_SEG_NAME="$(basename "$(dirname "${DP_SKILLSET_DIR}")")"
    DP_SEG_PREFIX="seg_${DP_OUTPUT_POLICY}_ck${DP_OUTPUT_CHECKPOINT}"
    if [[ "${DP_SEG_NAME}" = "${DP_SEG_PREFIX}"* ]]; then
      DP_SUFFIX="${DP_SEG_NAME#${DP_SEG_PREFIX}}"
    else
      DP_SUFFIX="_${DP_MODE}"
      if [ "${DP_THRESHOLD_MODE}" = "global_mean" ]; then
        DP_THRESHOLD_SCOPE="globalmean"
      else
        DP_THRESHOLD_SCOPE="episodemean"
      fi
      DP_THRESHOLD_PERCENT=$("${BOOTSTRAP_PYTHON}" -c 'import sys; v=float(sys.argv[1]); print(f"{v*100:g}".replace(".","p")+"p")' "${DP_THRESHOLD_SCALE}")
      DP_SUFFIX+="_${DP_THRESHOLD_SCOPE}_${DP_THRESHOLD_PERCENT}"
    fi
  fi
  # Shallow folder + DP-named HTML so different DPs don't overwrite each other (no deep nesting).
  # Strip the redundant dataset prefix from the run name (libero_90_full_full_state_obs10 → state_obs10).
  DP_TAG="${DP_OUTPUT_POLICY#${DP_OUTPUT_DATASET}_}"
  DP_OUT_DIR="${EVAL_DIR}/outputs/dp_skillset/${DP_OUTPUT_DATASET}"
  DP_EVAL_FILE_SUFFIX=""
  [ -n "${DP_EVAL_OUTPUT_SUFFIX}" ] && DP_EVAL_FILE_SUFFIX="_${DP_EVAL_OUTPUT_SUFFIX}"
  DP_OUT_HTML="${DP_TAG}_ck${DP_OUTPUT_CHECKPOINT}${DP_SUFFIX}${DP_EVAL_FILE_SUFFIX}.html"
  DP_TASK_ARGS=()
  [ -n "${DP_EVAL_TASK_IDS}" ] && DP_TASK_ARGS=(--task_ids ${DP_EVAL_TASK_IDS})
  DP_VIDEO_ARGS=()
  [ "${DP_EVAL_SKILL_VIDEO}" = "true" ] && DP_VIDEO_ARGS=(--skill_video)
  DP_DISPLAY_ARGS=()
  [ "${DP_EVAL_SHOW_START_END_FRAMES}" != "true" ] && DP_DISPLAY_ARGS+=(--hide_start_end_frames)
  [ "${DP_EVAL_SHOW_COS_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide_cos_graph)
  [ "${DP_EVAL_SHOW_GRIPPER_GRAPH}" != "true" ] && DP_DISPLAY_ARGS+=(--hide_gripper_graph)
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
      ${DP_VIDEO_ARGS[@]+"${DP_VIDEO_ARGS[@]}"} \
      ${DP_DISPLAY_ARGS[@]+"${DP_DISPLAY_ARGS[@]}"} \
      ${DP_TASK_ARGS[@]+"${DP_TASK_ARGS[@]}"}
  echo "DONE DP → ${DP_OUT_DIR}/${DP_OUT_HTML}"
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
      --dino_model_path     "${FSQ_EVAL_DINO_MODEL_PATH}" \
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
