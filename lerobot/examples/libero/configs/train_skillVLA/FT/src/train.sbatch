#!/usr/bin/env bash
# Fine-tune a complete Stage-2 checkpoint on a new SkillVLA dataset.
#SBATCH --job-name=stage2_ft
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
SUBMIT_DIR="${FT_TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${FT_TRAIN_CONFIG:-${SUBMIT_DIR}/ft_train_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${FT_ENV_SNAPSHOT:-}" ] && [ -f "${FT_ENV_SNAPSHOT}" ]; then
  source "${FT_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/ft_train_config.py" --config "${CONFIG_PATH}" --shell)"
fi
INITIALIZATION_MODE="${INITIALIZATION_MODE:-stage2}"
DSBC_LATENT_PREDICTOR_VLM_TOKENS="${DSBC_LATENT_PREDICTOR_VLM_TOKENS:-image_language}"
DSBC_NOISE_VLM_TOKENS="${DSBC_NOISE_VLM_TOKENS:-none}"
DSBC_LATENT_PREDICTOR_ENABLED="${DSBC_LATENT_PREDICTOR_ENABLED:-false}"
DSBC_SKILL_PREDICTOR_ENABLED="${DSBC_SKILL_PREDICTOR_ENABLED:-false}"
SKILL_PREDICTOR_ALL_LAYERS="${SKILL_PREDICTOR_ALL_LAYERS:-false}"
SKILL_PREDICTOR_LORA="${SKILL_PREDICTOR_LORA:-false}"
DSBC_SKILL_PREDICTOR_FREEZE_LORA="${DSBC_SKILL_PREDICTOR_FREEZE_LORA:-false}"

# With no Stage-2 checkpoint, the FT resolver emits the complete, validated
# Stage-2 environment from the inline direct_stage1 DSBC settings and replaces
# only the dataset/output fields. Reuse the canonical Stage-2 launcher so
# direct FT can never drift from Stage-2's policy arguments.
if [ "${INITIALIZATION_MODE}" = stage1_direct ]; then
  if [ -n "${FT_ENV_SNAPSHOT:-}" ] && [ -f "${FT_ENV_SNAPSHOT}" ]; then
    export STAGE2_ENV_SNAPSHOT="${FT_ENV_SNAPSHOT}"
  else
    # All resolver variables above are already exported by print_shell. An
    # existing empty file tells the shared launcher not to resolve a second
    # configuration.
    export STAGE2_ENV_SNAPSHOT=/dev/null
  fi
  exec bash \
    "${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skillVLA/stage2/src/train.sbatch"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
unset LD_LIBRARY_PATH

echo "START STAGE-2 FT (${STAGE2_MODE})"
echo "dataset  : ${SKILLVLA_DATASET_DIR}"
if [ "${INITIALIZATION_MODE}" = stage1 ]; then
  echo "recipe   : ${STAGE2_CHECKPOINT_PATH} (configuration only)"
  echo "prior    : ${POLICY_STAGE1_CHECKPOINT_PATH}"
else
  echo "parent   : ${STAGE2_CHECKPOINT_PATH} (weights + configuration)"
fi
echo "skill    : ${TRAINING_SKILL_SOURCE}"
echo "train    : ${FT_TRAIN_SCOPE}"
echo "noise    : mode=${DSBC_NOISE_OUTPUT_MODE}, VLM=${DSBC_NOISE_VLM_TOKENS}"
if [ "${DSBC_LATENT_PREDICTOR_ENABLED}" = true ]; then
  echo "latent   : mode=${DSBC_LATENT_PREDICTOR_MODE}, VLM=${DSBC_LATENT_PREDICTOR_VLM_TOKENS}, LoRA=${DSBC_LATENT_PREDICTOR_LORA}"
fi
if [ "${DSBC_SKILL_PREDICTOR_ENABLED}" = true ]; then
  echo "skill-p  : all_layers=${SKILL_PREDICTOR_ALL_LAYERS}, LoRA=${SKILL_PREDICTOR_LORA}, freeze_LoRA=${DSBC_SKILL_PREDICTOR_FREEZE_LORA}, hard=${DSBC_SKILL_HARD_WEIGHT}, STE=${DSBC_SKILL_STE_WEIGHT}, t=${DSBC_SKILL_TIMESTEPS}, samples=${DSBC_SKILL_SAMPLES_PER_SKILL}"
fi
echo "frozen   : complete Stage-1 VSA + base VLM"
echo "schedule : ${SCHEDULER_MODE} warmup=${SCHEDULER_WARMUP_STEPS}"
echo "output   : ${PT_OUTPUT_DIR}"
echo "dino     : ${POLICY_DINO_MODEL_PATH}"
echo "tokenizer: ${POLICY_TOKENIZER_PATH}"
echo "base VLM : ${POLICY_VLM_BASE_PATH}"
nvidia-smi || true

if [ ! -d "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing FT dataset: ${SKILLVLA_DATASET_DIR}" >&2
  exit 1
fi

CONFIG_JSON="${PT_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CONFIG_JSON}" ]; then
  echo "FT checkpoint found; resuming from ${CONFIG_JSON}"
  TRAIN_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
else
  if [ -e "${PT_OUTPUT_DIR}" ]; then
    # A startup failure (for example, an invalid checkpoint-owned model path)
    # leaves only W&B metadata behind. Preserve that evidence, but do not let it
    # permanently poison this run name.
    FAILED_RUNS_DIR="$(dirname "${PT_OUTPUT_DIR}")/.failed_runs"
    FAILED_RUN_NAME="$(basename "${PT_OUTPUT_DIR}").$(date +%Y%m%d_%H%M%S).${SLURM_JOB_ID:-$$}"
    mkdir -p "${FAILED_RUNS_DIR}"
    if [ -e "${FAILED_RUNS_DIR}/${FAILED_RUN_NAME}" ]; then
      echo "Failed-run archive already exists: ${FAILED_RUNS_DIR}/${FAILED_RUN_NAME}" >&2
      exit 1
    fi
    mv -- "${PT_OUTPUT_DIR}" "${FAILED_RUNS_DIR}/${FAILED_RUN_NAME}"
    echo "Archived non-resumable output -> ${FAILED_RUNS_DIR}/${FAILED_RUN_NAME}"
  fi
  if [ "${INITIALIZATION_MODE}" = stage1 ]; then
    # Load the old train_config as a recipe only. resume=false and the absence
    # of --policy.path cause a fresh Stage-2 construction from the Stage-1 path
    # recorded inside that config; no trained Stage-2 tensors are restored.
    TRAIN_ARGS=(
      --config_path="${STAGE2_TRAIN_CONFIG_PATH}"
      --resume=false
      --dataset.repo_id="${REPO_ID}"
      --dataset.root="${SKILLVLA_DATASET_DIR}"
      --dataset.video_backend=pyav
      --output_dir="${PT_OUTPUT_DIR}"
      --job_name="${PT_RUN_NAME}"
      --wandb.run_id=null
    )
  else
    TRAIN_ARGS=(
      --dataset.repo_id="${REPO_ID}"
      --dataset.root="${SKILLVLA_DATASET_DIR}"
      --dataset.video_backend=pyav
      --policy.path="${STAGE2_CHECKPOINT_PATH}"
      --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
      --policy.optimizer_lr="${LR}"
      --policy.scheduler_mode="${SCHEDULER_MODE}"
      --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
      --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
      --policy.device=cuda
      --policy.dtype=bfloat16
      --policy.push_to_hub=false
      --policy.compile_model=false
      --output_dir="${PT_OUTPUT_DIR}"
      --job_name="${PT_RUN_NAME}"
      --batch_size="${BATCH_SIZE}"
      --num_workers="${NUM_WORKERS}"
      --steps="${STEPS}"
      --log_freq="${LOG_FREQ}"
      --save_freq="${SAVE_FREQ}"
      --wandb.enable="${WANDB_ENABLE}"
      --wandb.project="${WANDB_PROJECT}"
    )
  fi
fi

# Historical checkpoints can retain absolute paths from the machine that
# created them. The resolver rebases those paths to this PROJECT_ROOT; pass the
# values as CLI overrides so they apply both to a fresh warm-start and resume.
POLICY_PATH_ARGS=()
[ -z "${POLICY_DINO_MODEL_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.dino_model_path="${POLICY_DINO_MODEL_PATH}")
[ -z "${POLICY_TOKENIZER_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.tokenizer_path="${POLICY_TOKENIZER_PATH}")
[ -z "${POLICY_VLM_BASE_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.vlm_base_path="${POLICY_VLM_BASE_PATH}")
[ -z "${POLICY_STAGE1_CHECKPOINT_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.stage1_checkpoint_path="${POLICY_STAGE1_CHECKPOINT_PATH}")
[ -z "${POLICY_SKILL_PREDICTOR_CHECKPOINT_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.skill_predictor_checkpoint_path="${POLICY_SKILL_PREDICTOR_CHECKPOINT_PATH}")
[ -z "${POLICY_FSQ_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.fsq_path="${POLICY_FSQ_PATH}")
[ -z "${POLICY_TERMINATOR_DINO_MODEL_PATH}" ] || \
  POLICY_PATH_ARGS+=(--policy.terminator_dino_model_path="${POLICY_TERMINATOR_DINO_MODEL_PATH}")

source "${PROJECT_ROOT}/lerobot/examples/libero/configs/gpu_guard.sh"
prepare_inline_cuda_guard
if PYTORCH_ALLOC_CONF=expandable_segments:True \
  accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" "${TRAIN_ARGS[@]}" "${POLICY_PATH_ARGS[@]}"; then
  TRAIN_STATUS=0
else
  TRAIN_STATUS=$?
fi
handle_inline_cuda_guard_exit "${TRAIN_STATUS}"
if (( TRAIN_STATUS != 0 )); then
  exit "${TRAIN_STATUS}"
fi

echo "DONE -> ${PT_OUTPUT_DIR}"
