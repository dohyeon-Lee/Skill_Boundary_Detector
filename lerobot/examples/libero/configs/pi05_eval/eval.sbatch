#!/usr/bin/env bash
#SBATCH --job-name=pi05_eval

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

# SBATCH --partition=debug
# SBATCH --nodelist=node200
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
[ ! -f "${CONFIG_PY}" ] && CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
  echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
  echo "Submit from lerobot root or from examples/libero/configs/pi05_eval." >&2
  exit 1
fi
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell | grep -E '^export (HOMEDIR|PROJDIR)=')"

cd ${HOMEDIR}${PROJDIR}/lerobot
source ${HOMEDIR}${PROJDIR}/.venv/bin/activate
unset LD_LIBRARY_PATH

export WANDB_CACHE_DIR=${HOMEDIR}/.wandb_cache
export WANDB_DATA_DIR=${HOMEDIR}/.wandb_data
mkdir -p ${WANDB_CACHE_DIR} ${WANDB_DATA_DIR}
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# ── Evaluation target ────────────────────────────────────────────────────────
TARGET_TASK=${TARGET_TASK:-libero_10}
TASK_IDS=${TASK_IDS:-"[0,1,2,3,4,5,6,7,8,9]"}
N_EPISODES=${N_EPISODES:-5}
EPISODE_OFFSET=${EPISODE_OFFSET:-25}   # 0 = 앞쪽부터, e.g. 45 = 에피소드 45번부터 (총 50개면 끝 5개)
N_ACTION_STEPS=${N_ACTION_STEPS:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}

# ── Checkpoint naming ────────────────────────────────────────────────────────
# PT train output:
#   ${HOMEDIR}${PROJDIR}/VLA_outputs/PT_${PT_DATASET}_pi05_batch${PT_BATCH_SIZE}_${PT_EXP}
# FT train output:
#   ${HOMEDIR}${PROJDIR}/VLA_outputs/FT_${PT_CHECKPOINT}PT_${FT_DATASET}_pi05_batch${FT_BATCH_SIZE}_${FT_EXP}
#
# Set MODEL directly to evaluate any custom run.
VLA_OUTPUT_ROOT=${HOMEDIR}${PROJDIR}/VLA_outputs
EVAL_STAGE=${EVAL_STAGE:-FT}  # PT or FT
CHECKPOINT=${CHECKPOINT:-003000}

PT_DATASET=${PT_DATASET:-libero_90}
PT_BATCH_SIZE=${PT_BATCH_SIZE:-32}
PT_EXP=${PT_EXP:-exp1}

FT_DATASET=${FT_DATASET:-libero_10_op1_10}
FT_BATCH_SIZE=${FT_BATCH_SIZE:-32}
FT_EXP=${FT_EXP:-exp1}
PT_CHECKPOINT=${PT_CHECKPOINT:-050000}

if [ -z "${MODEL:-}" ]; then
  if [ "${EVAL_STAGE}" = "FT" ]; then
    MODEL=FT_${PT_CHECKPOINT}PT_${FT_DATASET}_pi05_batch${FT_BATCH_SIZE}
    if [ -n "${FT_EXP}" ]; then
      MODEL=${MODEL}_${FT_EXP}
    fi
  else
    MODEL=PT_${PT_DATASET}_pi05_batch${PT_BATCH_SIZE}
    if [ -n "${PT_EXP}" ]; then
      MODEL=${MODEL}_${PT_EXP}
    fi
  fi
fi
POLICY_PATH=${POLICY_PATH:-${VLA_OUTPUT_ROOT}/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model}
if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi

MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-1}
MAX_VIDEOS_PER_TASK=${MAX_VIDEOS_PER_TASK:-1}
VIDEO_FRAME_STRIDE=${VIDEO_FRAME_STRIDE:-2}
VIDEO_FPS=${VIDEO_FPS:-10}

WANDB_PROJECT=VLA_eval
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${MODEL}_${CHECKPOINT}_${TARGET_TASK}}

# ── Output directory ─────────────────────────────────────────────────────────
if [ -d "${SUBMIT_DIR}/examples/libero/configs/pi05_eval" ]; then
  _EVAL_BASE="${SUBMIT_DIR}/examples/libero/configs/pi05_eval"
else
  _EVAL_BASE="${SUBMIT_DIR}"
fi
EVAL_OUT_DIR=${EVAL_OUT_DIR:-${_EVAL_BASE}/outputs/${WANDB_RUN_NAME}_offset${EPISODE_OFFSET}}
mkdir -p "${EVAL_OUT_DIR}"

echo "pi05 eval"
echo "  target     : ${TARGET_TASK}"
echo "  task_ids   : ${TASK_IDS}"
echo "  episodes   : ${N_EPISODES}"
echo "  videos     : ${MAX_VIDEOS_PER_TASK} per task, stride=${VIDEO_FRAME_STRIDE}, fps=${VIDEO_FPS}"
echo "  policy     : ${POLICY_PATH}"
echo "  output     : ${EVAL_OUT_DIR}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-eval \
  --policy.path=${POLICY_PATH} \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --eval.batch_size=${EVAL_BATCH_SIZE} \
  --eval.n_episodes=${N_EPISODES} \
  --eval.max_videos_per_task=${MAX_VIDEOS_PER_TASK} \
  --eval.video_frame_stride=${VIDEO_FRAME_STRIDE} \
  --eval.video_fps=${VIDEO_FPS} \
  --policy.device=cuda \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --env.task_ids=${TASK_IDS} \
  --env.init_state_offset=${EPISODE_OFFSET} \
  --env.max_parallel_tasks=${MAX_PARALLEL_TASKS} \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --output_dir="${EVAL_OUT_DIR}" \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

# ── Success rate chart ────────────────────────────────────────────────────────
PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  python - <<PYEOF
import json, sys
from pathlib import Path
import numpy as np

out_dir = Path("${EVAL_OUT_DIR}")
info_path = out_dir / "eval_info.json"
if not info_path.exists():
    print("eval_info.json not found, skipping chart.", file=sys.stderr)
    sys.exit(0)

info = json.loads(info_path.read_text())
per_task = info.get("per_task", [])
if not per_task:
    # flat structure fallback
    per_task = [{"task_id": i, "task_group": "task", "metrics": v}
                for i, v in enumerate(info.values()) if isinstance(v, dict)]

task_labels, task_success_rates = [], []
for t in sorted(per_task, key=lambda x: x.get("task_id", 0)):
    succs = t.get("metrics", {}).get("successes", [])
    rate = float(np.mean(succs)) if succs else 0.0
    task_labels.append(f"task{t.get('task_id', 0):02d}")
    task_success_rates.append(rate)

if not task_labels:
    print("No per-task data found, skipping chart.", file=sys.stderr)
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

height = max(2.8, 0.38 * len(task_labels) + 0.9)
fig, ax = plt.subplots(figsize=(8.0, height))
y = np.arange(len(task_labels))
values = np.nan_to_num(np.asarray(task_success_rates, dtype=np.float32), nan=0.0)
ax.barh(y, values, color="#4C78A8")
ax.set_yticks(y)
ax.set_yticklabels(task_labels)
ax.invert_yaxis()
ax.set_xlim(0.0, 1.0)
ax.set_xlabel("success rate")
ax.grid(axis="x", alpha=0.25)
for yi, value in zip(y, values):
    ax.text(min(value + 0.02, 0.98), yi, f"{value:.2f}", va="center", fontsize=8)
fig.tight_layout()
chart_path = out_dir / "task_success_rates.png"
fig.savefig(chart_path, dpi=160)
plt.close(fig)
print(f"Saved success rate chart -> {chart_path}")
PYEOF

echo "DONE → ${EVAL_OUT_DIR}"
