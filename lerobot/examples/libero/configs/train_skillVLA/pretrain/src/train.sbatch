#!/usr/bin/env bash
# SkillVLA autoregressive skill+FAST pretraining.
#SBATCH --job-name=VLApre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SUBMIT_DIR}/src"
CONFIG_PATH="${PRETRAIN_CONFIG:-${SUBMIT_DIR}/pretrain_config.yaml}"
BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3

if [ -n "${PRETRAIN_ENV_SNAPSHOT:-}" ] && [ -f "${PRETRAIN_ENV_SNAPSHOT}" ]; then
  source "${PRETRAIN_ENV_SNAPSHOT}"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/pretrain_config.py" --config "${CONFIG_PATH}" --shell)"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
unset LD_LIBRARY_PATH

echo "START SkillVLA FAST pretraining"
echo "Node      : ${SLURMD_NODENAME:-local}"
echo "mode      : ${TRAINING_MODE}"
echo "dataset   : ${SKILLVLA_DATASET_DIR}"
echo "segments  : ${TRANSITION_PACK} (randomization=${TRANSITION_RANDOMIZATION})"
echo "pi base   : ${PI_BASE}"
echo "FAST      : ${FAST_TOKENIZER} (vocab=${FAST_VOCAB_SIZE}, variable horizon)"
echo "targets   : ${PRETRAIN_TARGETS}"
echo "FSQ       : ${SKILL_FSQ_LEVELS}"
echo "output    : ${PRETRAIN_OUTPUT_DIR}  lr/bs=${LR}/${BATCH_SIZE}"
nvidia-smi || true

for required in "${SKILLVLA_DATASET_DIR}" "${TRANSITION_PACK}" "${PRETRAIN_TARGETS}" \
                "${PI_BASE}" "${TEXT_TOKENIZER}" "${FAST_TOKENIZER}"; do
  if [ ! -e "${required}" ]; then
    echo "Missing pretraining prerequisite: ${required}" >&2
    if [ "${required}" = "${FAST_TOKENIZER}" ] || [ "${required}" = "${PRETRAIN_TARGETS}" ]; then
      echo "Run ./prepare_fast_tokenizer.sh first (it needs the lerobot/fast-action-tokenizer source once)." >&2
    fi
    exit 1
  fi
done

CONFIG_JSON="${PRETRAIN_OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_ARGS=()
if [ -f "${CONFIG_JSON}" ]; then
  echo "Checkpoint found -> resuming from ${CONFIG_JSON}"
  RESUME_ARGS=(--resume=true --config_path="${CONFIG_JSON}")
elif [ -d "${PRETRAIN_OUTPUT_DIR}" ]; then
  echo "No checkpoint -> removing stale ${PRETRAIN_OUTPUT_DIR}"
  rm -rf "${PRETRAIN_OUTPUT_DIR}"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes="${NUM_GPUS}" \
  "${PROJECT_ROOT}/.venv/bin/lerobot-train" \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${SKILLVLA_DATASET_DIR}" \
    --dataset.video_backend=pyav \
    --policy.type=skill_vla_pretrain \
    --policy.pretrained_path="${PI_BASE}" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.training_mode="${TRAINING_MODE}" \
    --policy.text_tokenizer_name="${TEXT_TOKENIZER}" \
    --policy.action_tokenizer_name="${FAST_TOKENIZER}" \
    --policy.chunk_size=1 \
    --policy.n_action_steps=1 \
    --policy.max_state_dim="${MAX_STATE_DIM}" \
    --policy.max_action_dim="${MAX_ACTION_DIM}" \
    --policy.max_action_tokens="${MAX_FAST_TOKENS}" \
    --policy.fast_vocab_size="${FAST_VOCAB_SIZE}" \
    --policy.fast_skip_tokens="${FAST_SKIP_TOKENS}" \
    --policy.transition_packs="${TRANSITION_PACK}" \
    --policy.pretrain_target_packs="${PRETRAIN_TARGETS}" \
    --policy.transition_randomization="${TRANSITION_RANDOMIZATION}" \
    --policy.skill_fsq_levels="${SKILL_FSQ_LEVELS}" \
    --policy.skill_loss_weight="${SKILL_LOSS_WEIGHT}" \
    --policy.fast_loss_weight="${FAST_LOSS_WEIGHT}" \
    --policy.structure_loss_weight="${STRUCTURE_LOSS_WEIGHT}" \
    --policy.pretrain_lora_targets="${LORA_TARGETS}" \
    --policy.pretrain_lora_rank="${LORA_RANK}" \
    --policy.pretrain_lora_alpha="${LORA_ALPHA}" \
    --policy.pretrain_lora_dropout="${LORA_DROPOUT}" \
    --policy.optimizer_lr="${LR}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.scheduler_decay_lr="$("${BOOTSTRAP_PYTHON}" -c "print(${LR} * 0.1)")" \
    --output_dir="${PRETRAIN_OUTPUT_DIR}" \
    --job_name="${PRETRAIN_RUN_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    "${RESUME_ARGS[@]}"

echo "DONE -> ${PRETRAIN_OUTPUT_DIR}"
