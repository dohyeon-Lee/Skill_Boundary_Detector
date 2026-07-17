#!/usr/bin/env bash
# Inputs:
#   dataset name : TRAIN_DATA or target_dataset in train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
# Reference models/configs:
#   base DP cfg  : {project_root}/lerobot/{dp_base_config}
# Outputs:
#   DP policy    : {project_root}/outputs/DP/{dp_policy_name}
#   checkpoint   : {project_root}/outputs/DP/{dp_policy_name}/checkpoints/{dp_checkpoint}/pretrained_model
#
# Train Diffusion Policy (resnet: raw frames | state: proprioceptive-only).
#
# Submit through ../submit_train_dp.sh so yaml Slurm settings resolve first.
#
# Override dataset/config when needed:
#   TRAIN_DATA=libero_90_full_10 sbatch train_dp.sbatch
#   TRAIN_SKILLS_CONFIG=/path/to/train_skills_config.yaml sbatch train_dp.sbatch
#
#SBATCH --job-name=DP_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

# Slurm은 이 스크립트를 /var/spool/slurmd/<job>/ 로 복사해 실행하므로 BASH_SOURCE 기준
# 상대경로가 깨진다. 정상 경로는 submit_train_dp.sh가 전부 resolve+export한 env 상속 —
# 그 경우 재해석을 스킵. sbatch 단독 제출 폴백은 SLURM_SUBMIT_DIR(제출 cwd) 기준으로 복원.
if [ -n "${PROJECT_ROOT:-}" ] && [ -n "${RAW_DATASET_DIR:-}" ] && [ -n "${DP_OUTPUT_DIR:-}" ]; then
  echo "config resolved by submit wrapper (env inherited) — skip re-resolve"
else
  BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  CONFIG_PATH=${TRAIN_SKILLS_CONFIG:-${BASE_DIR}/dp_config.yaml}        # DP/ 에서 제출한 경우
  [ -f "${CONFIG_PATH}" ] || CONFIG_PATH="${BASE_DIR}/../dp_config.yaml"  # DP/src 에서 제출한 경우
  [ -f "${CONFIG_PATH}" ] || { echo "dp_config.yaml을 찾을 수 없음 (SLURM_SUBMIT_DIR=${BASE_DIR}; TRAIN_SKILLS_CONFIG로 절대경로 지정)" >&2; exit 1; }
  CONFIG_DIR="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)"
  COMMON_SRC_DIR="${CONFIG_DIR}/../src"                                  # configs/train_skills/src
  BOOTSTRAP_PYTHON="$(cd "${CONFIG_DIR}/../../../../../.." && pwd)/.venv/bin/python"
  [ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
  if [ -n "${TRAIN_DATA:-}" ]; then
    eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --dataset "${TRAIN_DATA}" --shell)"
  else
    eval "$("${BOOTSTRAP_PYTHON}" "${COMMON_SRC_DIR}/train_skills_config.py" --config "${CONFIG_PATH}" --shell)"
  fi
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"

echo "START DP TRAIN"
echo "Node       : ${SLURMD_NODENAME:-local}"
nvidia-smi

if [ ! -d "${RAW_DATASET_DIR}" ]; then
  echo "Dataset not found: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
if [ ! -f "${DP_BASE_CONFIG}" ]; then
  echo "DP base config not found: ${DP_BASE_CONFIG}" >&2
  exit 1
fi

RESUME_FLAG=""
CHECKPOINT_CONFIG="${DP_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CHECKPOINT_CONFIG}" ]; then
  RESUME_FLAG="--resume=true"
  TRAIN_CONFIG_PATH="${CHECKPOINT_CONFIG}"
elif [ -d "${DP_OUTPUT_DIR}" ]; then
  if [ "${DP_OVERWRITE_OUTPUT}" = "true" ]; then
    rm -rf "${DP_OUTPUT_DIR}"
    TRAIN_CONFIG_PATH="${DP_BASE_CONFIG}"
  else
    echo "Output dir already exists and no last checkpoint was found:" >&2
    echo "  ${DP_OUTPUT_DIR}" >&2
    echo "Set dp_overwrite_output: true in ${CONFIG_PATH} if you want to replace it." >&2
    exit 1
  fi
else
  TRAIN_CONFIG_PATH="${DP_BASE_CONFIG}"
fi

WANDB_ENABLE=${DP_WANDB_ENABLE}

# Vision encoder branch: "resnet" = original lerobot Diffusion Policy (ResNet on raw frames;
# image_features come from the dataset's camera keys). "state" = proprioceptive-only (no vision).
POLICY_ARGS=(
  --policy.n_obs_steps="${DP_N_OBS_STEPS}"
  --policy.n_action_steps="${DP_N_ACTION_STEPS}"
  --policy.horizon="${DP_HORIZON}"
)
if [ "${DP_VISION}" = "resnet" ]; then
  echo "DP train (vision=resnet, ORIGINAL Diffusion Policy — no DINO)"
  echo "  backbone     : ${DP_VISION_BACKBONE}"
  POLICY_ARGS+=(
    --policy.vision_backbone="${DP_VISION_BACKBONE}"
  )
elif [ "${DP_VISION}" = "state" ]; then
  echo "DP train (vision=state, proprioceptive-only — no image/DINO input)"
  POLICY_ARGS+=(
    --policy.state_only=true
  )
else
  echo "dp_vision must be 'resnet' or 'state' (got '${DP_VISION}') — DINO encoder 옵션은 은퇴함." >&2
  exit 1
fi

# RELATIVE actions (ABC/bimanual): action−state(anchor) 학습 — VLA/SBD probe와 동일 공간.
# 정규화는 relative 분포 stats(meta/relative_action_stats.json, build ④-b)를 사용.
if [ "${DP_RELATIVE:-false}" = "true" ]; then
  REL_STATS="${RAW_DATASET_DIR}/meta/relative_action_stats.json"
  if [ ! -f "${REL_STATS}" ]; then
    echo "dp_relative=true인데 relative stats 없음: ${REL_STATS}" >&2
    echo "  → ABC_dataset build ④-b(compute_relative_action_stats.py)를 먼저 실행하세요." >&2
    exit 1
  fi
  echo "  relative     : ON (stats: ${REL_STATS})"
  POLICY_ARGS+=(
    --policy.use_relative_actions=true
    --policy.relative_stats_path="${REL_STATS}"
  )
fi
echo "  dataset      : ${TARGET_DATASET}"
echo "  output       : ${DP_OUTPUT_DIR}"
echo "  config       : ${TRAIN_CONFIG_PATH}"
echo "  policy       : ${DP_POLICY}"

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  lerobot-train \
  --config_path="${TRAIN_CONFIG_PATH}" \
  --output_dir="${DP_OUTPUT_DIR}" \
  --job_name="${DP_POLICY}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${DP_WANDB_PROJECT}" \
  --steps="${DP_STEPS}" \
  --log_freq="${DP_LOG_FREQ}" \
  --save_freq="${DP_SAVE_FREQ}" \
  --eval_freq="${DP_EVAL_FREQ}" \
  --seed="${DP_SEED}" \
  "${POLICY_ARGS[@]}" \
  --dataset.repo_id="lerobot/${TARGET_DATASET}" \
  --dataset.root="${RAW_DATASET_DIR}" \
  --dataset.video_backend=pyav \
  --batch_size="${DP_BATCH_SIZE}" \
  --num_workers="${DP_NUM_WORKERS}" \
  ${RESUME_FLAG}

echo "DONE"
