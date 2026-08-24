#!/usr/bin/env bash
# pi05 closed-loop eval on LIBERO sim (lerobot-eval).
#   SINGLE model → rollout into EVAL_OUT_DIR (+ task_success_rates.png).
#   MULTI models (MODELS_JSON) → each model into panels/{label}/, per-task videos stitched into ONE
#   labelled grid task-by-task (side_by_side/). Task-split (eval_num_gpus>1) writes per-chunk
#   eval_info_{tag}.json → merge_eval_chunks.py combines at each chunk's end (idempotent).
# Config: ./pi05_eval_config.yaml (resolved by ../src/train_pi05_config.py).
#SBATCH --job-name=pi05_eval
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"             # .../train_pi05/pi05_eval
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"           # .../train_pi05
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${TRAIN_PI05_CONFIG:-${SUBMIT_DIR}/pi05_eval_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

# Prefer the submit-time env snapshot (frozen resolved env) → the job never re-runs the emitter on a
# live/edited/deleted yaml at start. Fallback to a fresh resolve only when run directly (no snapshot).
if [ -n "${PI05_EVAL_ENV_SNAPSHOT:-}" ] && [ -f "${PI05_EVAL_ENV_SNAPSHOT}" ]; then
  source "${PI05_EVAL_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"
fi

# JOB-ARRAY fan-out (submit_eval.sh submits ONE --array job → tasks show as {ID}_0,{ID}_1,… and
# `scancel {ID}` kills them all): each array task picks its own line from EVAL_FANOUT
# ("PANEL_LABEL|TASK_IDS|TASK_TAG" per line, index = SLURM_ARRAY_TASK_ID). Applied AFTER sourcing the
# env snapshot so THIS chunk's task_ids wins — the snapshot exports the FULL task_ids, so a pre-source
# override would be clobbered (every array task would then run ALL tasks, not its chunk).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_FANOUT:-}" ]; then
  _LINE="$(printf '%s\n' "${EVAL_FANOUT}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
  IFS='|' read -r _PL _TI _TT <<< "${_LINE}"
  [ -n "${_PL}" ] && export PANEL_LABEL="${_PL}"
  [ -n "${_TI}" ] && export EVAL_TASK_IDS="${_TI}"
  [ -n "${_TT}" ] && export TASK_TAG="${_TT}"     # lerobot_eval reads TASK_TAG → eval_info_{tag}.json
  echo "array task ${SLURM_ARRAY_TASK_ID}: panel='${_PL:-<all>}' task_ids=${_TI} tag='${_TT:-}'"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

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


echo "START pi05 eval"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "target    : ${EVAL_TARGET_TASK}  task_ids=${EVAL_TASK_IDS}  episodes=${EVAL_N_EPISODES}/task"
echo "mode      : $([ "${EVAL_GRAPH_ONLY}" = "true" ] && echo 'graph-only (no videos)' || echo 'full')"
nvidia-smi || true
mkdir -p "${EVAL_OUT_DIR}"
BASE_RN="${EVAL_WANDB_RUN_NAME}"
[ -n "${TASK_TAG:-}" ] && BASE_RN="${EVAL_WANDB_RUN_NAME%_${TASK_TAG}}"

# run_eval OUT JOB POLICY ONTASK — one lerobot-eval rollout (shared knobs from the emitter env; per-model
# policy + output are the arguments; ONTASK = --eval.on_task_done_cmd for progressive per-task stitch).
run_eval() {
  local out="$1" job="$2" ppath="$3" ontask="${4:-}"
  local policy_n_action_steps
  policy_n_action_steps="$("${PYTHON_BIN}" - "${ppath}/config.json" "${EVAL_N_ACTION_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
default_steps = int(sys.argv[2])
config = json.loads(config_path.read_text()) if config_path.is_file() else {}
enabled = str(config.get("use_eef_relative_actions", False)).strip().lower() in {
    "1", "true", "yes", "on"
}
print(1 if enabled else default_steps)
PY
)"
  mkdir -p "${out}"
  if [ "${policy_n_action_steps}" = "1" ] && [ "${EVAL_N_ACTION_STEPS}" != "1" ]; then
    echo "EEF-relative checkpoint: overriding n_action_steps=${EVAL_N_ACTION_STEPS} -> 1"
  fi
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" "${EVAL_BIN}" \
    --policy.path="${ppath}" \
    --policy.tokenizer_path="${PI05_TOKENIZER_PATH}" \
    --env.type=libero \
    --env.task="${EVAL_TARGET_TASK}" \
    --env.task_ids="${EVAL_TASK_IDS}" \
    --env.init_state_offset="${EVAL_EPISODE_OFFSET}" \
    --env.max_parallel_tasks="${EVAL_MAX_PARALLEL_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${EVAL_N_EPISODES}" \
    --eval.max_videos_per_task="${EVAL_MAX_VIDEOS_PER_TASK}" \
    --eval.video_frame_stride="${EVAL_VIDEO_FRAME_STRIDE}" \
    --eval.video_fps="${EVAL_VIDEO_FPS}" \
    --eval.on_task_done_cmd="${ontask}" \
    --policy.device=cuda \
    --policy.n_action_steps="${policy_n_action_steps}" \
    --wandb_project "${EVAL_WANDB_PROJECT}" \
    --job_name "${job}" \
    --output_dir="${out}" \
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'
}

# merge_chunks OUT JOB — combine per-chunk eval_info_{tag}.json → eval_info.json + chart (task-split).
merge_chunks() {
  "${PYTHON_BIN}" "${SRC_DIR}/merge_eval_chunks.py" \
    --out_dir="$1" --job_name="$2" \
    --expected_tasks="${TASKS_TOTAL:-0}" --wandb_project="${EVAL_WANDB_PROJECT}" \
    || echo "chunk merge failed (non-fatal)"
}

if [ -z "${MODELS_JSON:-}" ]; then
  # ── SINGLE model ──
  echo "policy    : ${EVAL_POLICY_PATH}"
  if [ ! -d "${EVAL_POLICY_PATH}" ]; then
    echo "Policy checkpoint not found: ${EVAL_POLICY_PATH}" >&2
    exit 1
  fi
  run_eval "${EVAL_OUT_DIR}" "${EVAL_WANDB_RUN_NAME}" "${EVAL_POLICY_PATH}"

  if [ -n "${TASK_TAG:-}" ]; then
    merge_chunks "${EVAL_OUT_DIR}" "${BASE_RN}"
  else
    # Single non-chunk job → the canonical task_success_rates.png inline (merge handles the chunk case).
    PYTHONPATH="${LEROBOT_ROOT}/src:${LIBERO_REPO_ROOT}" "${PYTHON_BIN}" - <<PYEOF || echo "chart skipped"
import json, sys
from pathlib import Path
import numpy as np
out_dir = Path("${EVAL_OUT_DIR}")
info_path = out_dir / "eval_info.json"
if not info_path.exists():
    sys.exit(0)
info = json.loads(info_path.read_text())
per_task = info.get("per_task", [])
labels, rates = [], []
for t in sorted(per_task, key=lambda x: x.get("task_id", 0)):
    s = t.get("metrics", {}).get("successes", [])
    labels.append(f"task{int(t.get('task_id', 0)):02d}")
    rates.append(float(np.mean(s)) if s else 0.0)
if not labels:
    sys.exit(0)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
height = max(2.8, 0.38 * len(labels) + 0.9)
fig, ax = plt.subplots(figsize=(8.0, height))
y = np.arange(len(labels))
ax.barh(y, np.nan_to_num(rates), color="#4C78A8")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlim(0.0, 1.0); ax.set_xlabel("success rate"); ax.grid(axis="x", alpha=0.25)
for yi, v in zip(y, rates):
    ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "task_success_rates.png", dpi=160)
print(f"Saved chart -> {out_dir/'task_success_rates.png'}")
PYEOF
  fi
else
  # ── MULTI-model side-by-side: each model over the SAME tasks/seeds into panels/{label}/, then stitch
  # per-task videos into ONE labelled grid → side_by_side/{task}/. ──
  echo "multi     : $("${BOOTSTRAP_PYTHON}" -c 'import json,os; print(", ".join(m["label"] for m in json.loads(os.environ["MODELS_JSON"])))')"
  SPECS="$("${BOOTSTRAP_PYTHON}" - <<'PY'
import json, os
for m in json.loads(os.environ["MODELS_JSON"]):
    print("|".join([m["label"], m["policy_path"], m.get("checkpoint", "")]))
PY
)"
  # Progressive per-task stitch for full eval only. Graph-only deliberately
  # avoids even launching the stitch helper because no videos are produced.
  STITCH_SH="${EVAL_OUT_DIR}/.stitch_panels.sh"
  STITCH_CALLBACK=""
  mkdir -p "${EVAL_OUT_DIR}"
  if [ "${EVAL_GRAPH_ONLY}" != "true" ]; then
    cat > "${STITCH_SH}" <<EOF
#!/usr/bin/env bash
"${PYTHON_BIN}" "${SRC_DIR}/stitch_panels.py" \
  --panels_root="${EVAL_OUT_DIR}/panels" --out_dir="${EVAL_OUT_DIR}/side_by_side" \
  --labels_json='${MODELS_LABELS}' --per_row="${MODELS_PER_ROW}"
EOF
    STITCH_CALLBACK="bash ${STITCH_SH}"
  fi

  # PANEL_LABEL set (submit's model×chunk fan-out) → this job runs ONLY that model; unset (srun / no
  # fan-out) → all models sequentially.
  while IFS='|' read -r LBL PPATH CKPT; do
    [ -z "${LBL}" ] && continue
    if [ -n "${PANEL_LABEL:-}" ] && [ "${LBL}" != "${PANEL_LABEL}" ]; then
      continue
    fi
    if [ ! -d "${PPATH}" ]; then
      echo "[warn] panel '${LBL}': checkpoint missing (${PPATH}) — skipping this model" >&2
      continue
    fi
    echo "── panel '${LBL}': ${PPATH}"
    run_eval "${EVAL_OUT_DIR}/panels/${LBL}" "${BASE_RN}_${LBL}${TASK_TAG:+_${TASK_TAG}}" \
             "${PPATH}" "${STITCH_CALLBACK}"
  done <<< "${SPECS}"

  # Stitch is IDEMPOTENT + ATOMIC + skips episodes not yet finished in every panel → run it at the end
  # of EVERY job; whichever job finishes a task LAST completes that task's grid clip immediately.
  if [ "${EVAL_GRAPH_ONLY}" != "true" ]; then
    "${PYTHON_BIN}" "${SRC_DIR}/stitch_panels.py" \
      --panels_root="${EVAL_OUT_DIR}/panels" --out_dir="${EVAL_OUT_DIR}/side_by_side" \
      --labels_json="${MODELS_LABELS}" --per_row="${MODELS_PER_ROW}" \
      || echo "stitch failed (non-fatal)"
  fi

  # Per-panel summary + success-rate chart for EVERY model (chunked → combine eval_info_t*.json;
  # non-chunked → from that panel's eval_info.json). With PANEL_LABEL only THIS job's panel is merged
  # (its other chunks may still be running — idempotent).
  for PDIR in "${EVAL_OUT_DIR}"/panels/*/; do
    [ -d "${PDIR}" ] || continue
    if [ -n "${PANEL_LABEL:-}" ] && [ "$(basename "${PDIR}")" != "${PANEL_LABEL}" ]; then
      continue
    fi
    merge_chunks "${PDIR%/}" "${BASE_RN}_$(basename "${PDIR}")"
  done
fi

echo "DONE → ${EVAL_OUT_DIR}"
