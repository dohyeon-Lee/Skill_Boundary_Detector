#!/usr/bin/env bash
# pi05 closed-loop eval.
#
# Inputs:
#   policy : {project_root}/{pi05_outputs_root}/{eval_model}/checkpoints/{eval_checkpoint}/pretrained_model
#   env    : LIBERO sim task {eval_target_task}
# Output:
#   eval   : ./outputs/{eval_wandb_run_name}_offset{eval_episode_offset}
# Config:
#   ../train_pi05_config.yaml
#SBATCH --job-name=pi05_eval
#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
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
ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"          # .../train_pi05
CONFIG_PATH="${TRAIN_PI05_CONFIG:-${ROOT_DIR}/train_pi05_config.yaml}"
CONFIG_PY="${ROOT_DIR}/src/train_pi05_config.py"

BOOTSTRAP_PYTHON="${ROOT_DIR}/../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --config "${CONFIG_PATH}" --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
unset LD_LIBRARY_PATH

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HOME}/.wandb_cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${HOME}/.wandb_data}"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START pi05 eval"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "target    : ${EVAL_TARGET_TASK}"
echo "task_ids  : ${EVAL_TASK_IDS}"
echo "episodes  : ${EVAL_N_EPISODES}"
echo "policy    : ${EVAL_POLICY_PATH}"
nvidia-smi || true

if [ ! -d "${EVAL_POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${EVAL_POLICY_PATH}" >&2
  exit 1
fi

EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT_DIR}/pi05_eval/outputs/${EVAL_WANDB_RUN_NAME}_offset${EVAL_EPISODE_OFFSET}}"
mkdir -p "${EVAL_OUT_DIR}"

PYTHONPATH="${LEROBOT_ROOT}/src" "${EVAL_BIN}" \
  --policy.path="${EVAL_POLICY_PATH}" \
  --env.type=libero \
  --env.task="${EVAL_TARGET_TASK}" \
  --eval.batch_size="${EVAL_BATCH_SIZE}" \
  --eval.n_episodes="${EVAL_N_EPISODES}" \
  --eval.max_videos_per_task="${EVAL_MAX_VIDEOS_PER_TASK}" \
  --eval.video_frame_stride="${EVAL_VIDEO_FRAME_STRIDE}" \
  --eval.video_fps="${EVAL_VIDEO_FPS}" \
  --policy.device=cuda \
  --policy.n_action_steps="${EVAL_N_ACTION_STEPS}" \
  --env.task_ids="${EVAL_TASK_IDS}" \
  --env.init_state_offset="${EVAL_EPISODE_OFFSET}" \
  --env.max_parallel_tasks="${EVAL_MAX_PARALLEL_TASKS}" \
  --wandb_project "${EVAL_WANDB_PROJECT}" \
  --job_name "${EVAL_WANDB_RUN_NAME}" \
  --output_dir="${EVAL_OUT_DIR}" \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

PYTHONPATH="${LEROBOT_ROOT}/src" "${PYTHON_BIN}" - <<PYEOF
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

echo "DONE -> ${EVAL_OUT_DIR}"
