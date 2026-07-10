#!/usr/bin/env bash
# SkillVLA FT closed-loop EVAL on LIBERO sim via lerobot_skillvla_eval.
# Rolls out a FINETUNED policy; the terminator (ft | base | stage2) gates skill transitions.
# train_terminator is forced false so no co-training terminator is built at eval.
#SBATCH --job-name=ft_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${FT_EVAL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"   # FT_eval (FT_EVAL_DIR passed by submit → correct under srun-in-salloc)
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FT_EVAL_CONFIG:-${SUBMIT_DIR}/ft_eval_config.yaml}"
# Shared helpers (generic; no stage2 coupling) live in stage2_eval/src — reuse instead of duplicating.
S2_SRC_DIR="${SUBMIT_DIR}/../stage2_eval/src"
EXPORT_TERM_PY="${SUBMIT_DIR}/../FT/src/export_ft_terminator.py"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

# Prefer the submit-time env snapshot (frozen resolved env) → the job never re-runs the emitter on a
# live/edited/deleted yaml at start. Fallback to a fresh resolve only when run directly (no snapshot).
if [ -n "${FT_EVAL_ENV_SNAPSHOT:-}" ] && [ -f "${FT_EVAL_ENV_SNAPSHOT}" ]; then
  source "${FT_EVAL_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_eval_config.py" --config "${CONFIG_PATH}" --shell)"
fi

# JOB-ARRAY fan-out (submit_eval.sh submits ONE --array job → tasks show as {ID}_0,{ID}_1,… and
# `scancel {ID}` kills them all): each array task picks its own line from EVAL_FANOUT
# ("PANEL_LABEL|TASK_IDS|TASK_TAG" per line, index = SLURM_ARRAY_TASK_ID). Applied AFTER sourcing the
# env snapshot so THIS chunk's TASK_IDS wins — the snapshot exports the FULL task_ids, so a pre-source
# override would be clobbered (every array task would then run ALL tasks, not its chunk).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_FANOUT:-}" ]; then
  _LINE="$(printf '%s\n' "${EVAL_FANOUT}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  IFS='|' read -r _PL _TI _TT <<< "${_LINE}"
  [ -n "${_PL}" ] && export PANEL_LABEL="${_PL}"
  [ -n "${_TI}" ] && export TASK_IDS="${_TI}"
  [ -n "${_TT}" ] && export TASK_TAG="${_TT}"
  echo "array task ${SLURM_ARRAY_TASK_ID}: panel='${_PL:-<all>}' task_ids=${_TI} tag='${_TT:-}'"
fi

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

# ── helpers ─────────────────────────────────────────────────────────────────────────────────
# resolve_fsq KIND BASE FT_PATH FT_RUN CKPT S2_PATH S2_RUN S2_CKPT → echoes the FSQ to eval with.
#   base   → BASE
#   ft     → FT_PATH (lazy-export checkpoints/<ckpt>/FSQ_ft.pt from FT_RUN@CKPT if missing; fall back
#            to BASE if the export produced nothing)
#   stage2 → S2_PATH (lazy-export <stage2_ckpt>/FSQ_ft.pt from S2_RUN@S2_CKPT if missing)
# Lazy exports are ATOMIC (per-job tmp then mv -n) so task-split array jobs don't race.
resolve_fsq() {
  local kind="$1" base="$2" ftp="$3" ftrun="$4" ckpt="$5" s2p="${6:-}" s2run="${7:-}" s2ckpt="${8:-}"
  case "${kind}" in
    base)
      echo "${base}" ;;
    stage2)
      if [ ! -f "${s2p}" ]; then
        echo "Stage-2 terminator missing → exporting from ${s2run} @ ${s2ckpt}" >&2
        local tmp="${s2p}.tmp.${SLURM_JOB_ID:-$$}"
        "${PROJECT_ROOT}/.venv/bin/python" "${EXPORT_TERM_PY}" \
          --ft_run_dir="${s2run}" --checkpoint="${s2ckpt}" \
          --base_fsq="${base}" --out="${tmp}" --skip_if_no_terminator >&2
        if [ -f "${tmp}" ]; then mv -n "${tmp}" "${s2p}" 2>/dev/null || true; rm -f "${tmp}"; fi
      fi
      echo "${s2p}" ;;
    *)  # ft
      if [ ! -f "${ftp}" ]; then
        echo "Adapted terminator missing for ${ckpt} → exporting from the checkpoint" >&2
        local tmp="${ftp}.tmp.${SLURM_JOB_ID:-$$}"
        "${PROJECT_ROOT}/.venv/bin/python" "${EXPORT_TERM_PY}" \
          --ft_run_dir="${ftrun}" --checkpoint="${ckpt}" \
          --base_fsq="${base}" --out="${tmp}" --skip_if_no_terminator >&2
        if [ -f "${tmp}" ]; then mv -n "${tmp}" "${ftp}" 2>/dev/null || true; rm -f "${tmp}"; fi
      fi
      if [ -f "${ftp}" ]; then echo "${ftp}"; else echo "${base}"; fi ;;
  esac
}

# run_eval OUT JOB POLICY FSQ GT_DIR SKILL_HTML INIT_STATES DROP_VLM — one lerobot_skillvla_eval
# rollout (shared knobs from the emitter env; per-model bits are the arguments; INIT_STATES ""=seed
# resets; DROP_VLM true = sever cond/action→VLM at inference, skill code only).
run_eval() {
  local out="$1" job="$2" ppath="$3" fsq="$4" gt="$5" shtml="$6" inits="${7:-}" dropvlm="${8:-false}" ontask="${9:-}" vlmovr="${10:-}"
  mkdir -p "${out}"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" \
    "${PROJECT_ROOT}/.venv/bin/python" -m lerobot.scripts.lerobot_skillvla_eval \
      --policy.path="${ppath}" \
      --policy.fsq_path="${fsq}" \
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
      --eval.skill_html="${shtml}" \
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
      --policy.gt_skill_dataset_dir="${gt}" \
      --policy.skill_advance_mode="${SKILL_ADVANCE_MODE}" \
      --policy.eval_drop_vlm="${dropvlm}" \
      --policy.eval_vlm_override_path="${vlmovr}" \
      --eval.init_states_path="${inits}" \
      --eval.on_task_done_cmd="${ontask}" \
      --wandb_project="${WANDB_PROJECT}" \
      --job_name="${job}" \
      --output_dir="${out}" \
      --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
      ${EXTRA_ARGS:-}
}

echo "START SkillVLA FT EVAL"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "target   : ${TARGET_TASK}  task_ids=${TASK_IDS}  episodes=${N_EPISODES}/task"
nvidia-smi || true
mkdir -p "${EVAL_OUT_DIR}"
BASE_RN="${WANDB_RUN_NAME}"
[ -n "${TASK_TAG:-}" ] && BASE_RN="${WANDB_RUN_NAME%_${TASK_TAG}}"

if [ -z "${MODELS_JSON:-}" ]; then
  # ── SINGLE model ──
  FSQ_FOR_EVAL="$(resolve_fsq "${TERM_KIND}" "${BASE_FSQ}" "${FT_FSQ_PATH}" "${FT_RUN_DIR}" \
                  "${CHECKPOINT}" "${STAGE2_FSQ_PATH}" "${STAGE2_RUN_DIR}" "${STAGE2_CHECKPOINT}")"
  echo "policy   : ${POLICY_PATH}"
  echo "fsq      : ${FSQ_FOR_EVAL}  (terminator=${TERM_KIND})"
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
  [ -n "${EVAL_INIT_STATES_PATH}" ] && echo "scenes   : EPISODE-EXACT (${EVAL_INIT_STATES_PATH})"
  [ -n "${VLM_OVERRIDE_PATH}" ] && echo "vlm      : Stage-2 override (${VLM_OVERRIDE_PATH})"
  run_eval "${EVAL_OUT_DIR}" "${WANDB_RUN_NAME}" "${POLICY_PATH}" "${FSQ_FOR_EVAL}" \
           "${GT_SKILL_DATASET_DIR}" "${SKILL_HTML_EFFECTIVE}" "${EVAL_INIT_STATES_PATH}" \
           "false" "" "${VLM_OVERRIDE_PATH}"

  if [ -n "${TASK_TAG:-}" ]; then
    "${PROJECT_ROOT}/.venv/bin/python" "${S2_SRC_DIR}/merge_eval_chunks.py" \
      --out_dir="${EVAL_OUT_DIR}" --target_task="${TARGET_TASK}" --job_name="${BASE_RN}" \
      --expected_tasks="${TASKS_TOTAL:-0}" --wandb_project="${WANDB_PROJECT}" \
      || echo "chunk merge failed (non-fatal)"
  fi
else
  # ── MULTI-model side-by-side: run each model over the SAME tasks/seeds into panels/{label}/,
  # then stitch the per-task videos into ONE labelled grid → side_by_side/{task}/. ──
  echo "multi    : $("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(", ".join(m["label"] for m in json.loads(os.environ["MODELS_JSON"])))')"
  SPECS="$("${BOOTSTRAP_PYTHON}" - <<'PY'
import json, os
for m in json.loads(os.environ["MODELS_JSON"]):
    print("|".join([m["label"], m["policy_path"], m["base_fsq"], m["ft_fsq_path"],
                    m["ft_run_dir"], m["checkpoint"], m["term_kind"], m["gt_skill_dataset_dir"],
                    m.get("eval_init_states_path", ""), str(m.get("drop_vlm", False)).lower(),
                    m.get("stage2_fsq_path", ""), m.get("stage2_run_dir", ""),
                    m.get("stage2_checkpoint", ""), m.get("vlm_override_path", "")]))
PY
)"
  # Progressive per-task stitch (like stage1): each panel eval runs this after EVERY task's videos are
  # written, so a task's grid clip appears as soon as it's done in all panels — instead of all-at-job-end.
  # A tiny helper script sidesteps nested shell quoting of the labels_json in --eval.on_task_done_cmd.
  STITCH_SH="${EVAL_OUT_DIR}/.stitch_panels.sh"
  mkdir -p "${EVAL_OUT_DIR}"
  cat > "${STITCH_SH}" <<EOF
#!/usr/bin/env bash
"${PROJECT_ROOT}/.venv/bin/python" "${S2_SRC_DIR}/stitch_panels.py" \
  --panels_root="${EVAL_OUT_DIR}/panels" --out_dir="${EVAL_OUT_DIR}/side_by_side" \
  --labels_json='${MODELS_LABELS}' --per_row="${MODELS_PER_ROW}"
EOF

  # PANEL_LABEL set (submit's model×chunk fan-out) → this job runs ONLY that model; unset (srun /
  # legacy) → all models sequentially.
  while IFS='|' read -r LBL PPATH BFSQ FTFSQ FTRUN CKPT KIND GTDIR INITS DROP S2FSQ S2RUN S2CKPT VLMOVR; do
    [ -z "${LBL}" ] && continue
    if [ -n "${PANEL_LABEL:-}" ] && [ "${LBL}" != "${PANEL_LABEL}" ]; then
      continue
    fi
    if [ ! -d "${PPATH}" ]; then
      echo "[warn] panel '${LBL}': checkpoint missing (${PPATH}) — skipping this model" >&2
      continue
    fi
    FSQ="$(resolve_fsq "${KIND}" "${BFSQ}" "${FTFSQ}" "${FTRUN}" "${CKPT}" "${S2FSQ}" "${S2RUN}" "${S2CKPT}")"
    if [ ! -f "${FSQ}" ]; then
      echo "[warn] panel '${LBL}': no terminator FSQ (${FSQ}) — skipping this model" >&2
      continue
    fi
    echo "── panel '${LBL}': ${PPATH}  (fsq=${FSQ}, terminator=${KIND}${INITS:+, scenes=EPISODE-EXACT}, drop_vlm=${DROP:-false}${VLMOVR:+, vlm=Stage-2})"
    run_eval "${EVAL_OUT_DIR}/panels/${LBL}" "${BASE_RN}_${LBL}${TASK_TAG:+_${TASK_TAG}}" \
             "${PPATH}" "${FSQ}" "${GTDIR}" false "${INITS}" "${DROP:-false}" "bash ${STITCH_SH}" "${VLMOVR}"
  done <<< "${SPECS}"

  # Stitch is IDEMPOTENT + ATOMIC + skips episodes not yet finished in every panel → run it at the end
  # of EVERY job; whichever job finishes a task LAST completes that task's grid clip immediately.
  "${PROJECT_ROOT}/.venv/bin/python" "${S2_SRC_DIR}/stitch_panels.py" \
    --panels_root="${EVAL_OUT_DIR}/panels" --out_dir="${EVAL_OUT_DIR}/side_by_side" \
    --labels_json="${MODELS_LABELS}" --per_row="${MODELS_PER_ROW}" \
    || echo "stitch failed (non-fatal)"

  # Task-split: merge each panel's chunks (summary json + full chart/index per panel). With
  # PANEL_LABEL, only THIS job's panel is merged (its other chunks may still be running — idempotent).
  if [ -n "${TASK_TAG:-}" ]; then
    for PDIR in "${EVAL_OUT_DIR}"/panels/*/; do
      [ -d "${PDIR}" ] || continue
      if [ -n "${PANEL_LABEL:-}" ] && [ "$(basename "${PDIR}")" != "${PANEL_LABEL}" ]; then
        continue
      fi
      "${PROJECT_ROOT}/.venv/bin/python" "${S2_SRC_DIR}/merge_eval_chunks.py" \
        --out_dir="${PDIR%/}" --target_task="${TARGET_TASK}" --job_name="${BASE_RN}_$(basename "${PDIR}")" \
        --expected_tasks="${TASKS_TOTAL:-0}" --wandb_project="${WANDB_PROJECT}" \
        || echo "chunk merge failed for $(basename "${PDIR}") (non-fatal)"
    done
  fi
fi

echo "DONE → ${EVAL_OUT_DIR}"
