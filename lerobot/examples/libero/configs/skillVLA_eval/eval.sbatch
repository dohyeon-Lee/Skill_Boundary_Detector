#!/usr/bin/env bash
#SBATCH --job-name=skillVLA_eval

#SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31,node28

# SBATCH --partition=debug
# SBATCH --exclude=node200
# SBATCH --nodelist=node100

#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
CONFIG_PY=${SUBMIT_DIR}/examples/libero/configs/data_generation/pipeline_config.py
if [ ! -f "${CONFIG_PY}" ]; then
  CONFIG_PY=${SUBMIT_DIR}/../data_generation/pipeline_config.py
fi
if [ ! -f "${CONFIG_PY}" ]; then
  echo "Could not find pipeline_config.py from ${SUBMIT_DIR}." >&2
  echo "Submit from lerobot root or from examples/libero/configs/skillVLA_eval." >&2
  exit 1
fi
BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../.venv/bin/python
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=${SUBMIT_DIR}/../../../../../.venv/bin/python
fi
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi
if [ -n "${EVAL_DATA:-}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell --data "${EVAL_DATA}")"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${CONFIG_PY}" --shell)"
fi

cd "${HOMEDIR}${PROJDIR}/lerobot"
source "${HOMEDIR}${PROJDIR}/.venv/bin/activate"

# Avoid system cuDNN shadowing the cuDNN bundled with the installed PyTorch.
unset LD_LIBRARY_PATH
CFG_SKILL_DECODER_STATE_INDICES=${SKILL_DECODER_STATE_INDICES}
CFG_SKILL_DECODER_PRIOR_NOISE_RATIO=${EVAL_SKILL_DECODER_PRIOR_NOISE_RATIO}
CFG_INFERENCE_SKILL_MAX_LENGTH=${INFERENCE_SKILL_MAX_LENGTH}
CFG_DETACH_ACTION_PREFIX_GRAD=${EVAL_DETACH_ACTION_PREFIX_GRAD}
CFG_TARGET_TASK=${TARGET_TASK}

export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${HOMEDIR}/.wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${HOMEDIR}/.wandb_data}
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"
export WANDB_DISABLE_SYMLINKS=true

echo "START JOB"
echo "Node: ${SLURMD_NODENAME:-local}"
nvidia-smi

# ── Evaluation target ────────────────────────────────────────────────────────
# Edit these per run. EVAL_DATA defaults to pipeline_config.yaml:data.
TARGET_TASK=${TARGET_TASK_OVERRIDE:-${CFG_TARGET_TASK}}
TASK_IDS=${TASK_IDS:-"[0,1,2,3,4,5,6,7,8,9]"}
N_EPISODES=${N_EPISODES:-1}
N_ACTION_STEPS=${N_ACTION_STEPS:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
MAX_PARALLEL_TASKS=${MAX_PARALLEL_TASKS:-1}
MAX_VIDEOS_PER_TASK=${MAX_VIDEOS_PER_TASK:-1}
VIDEO_FRAME_STRIDE=${VIDEO_FRAME_STRIDE:-2}
VIDEO_FPS=${VIDEO_FPS:-10}
SKILL_HTML=${SKILL_HTML:-true}
SKILL_HTML_TRAIN_SAMPLES=${SKILL_HTML_TRAIN_SAMPLES:-10}

# ── SkillVLA checkpoint ──────────────────────────────────────────────────────
# eval_model_type (from pipeline_config.yaml): PT | FT
# Override at submit time: sbatch --export=ALL,EVAL_MODEL_TYPE=FT,...
VLA_OUTPUT_ROOT=${HOMEDIR}${PROJDIR}/VLA_outputs
FSQ_TAG=fsq${FSQ_LEVELS_STR// /}

# PT default model
PT_BATCH_SIZE=${PT_BATCH_SIZE:-32}
PT_EXP=${PT_EXP:-${EVAL_EXP}}
PT_DEFAULT_MODEL=PT_${SKILLVLA_DATASET}_${FSQ_TAG}_FSQ_epoch${FSQ_EPOCH}_batch${PT_BATCH_SIZE}
if [ -n "${PT_EXP}" ]; then
  PT_DEFAULT_MODEL=${PT_DEFAULT_MODEL}_${PT_EXP}
fi

# FT default model (derived by pipeline_config.py → FT_EVAL_MODEL)
FT_DEFAULT_MODEL=${FT_EVAL_MODEL}

# Select model / checkpoint by type
EVAL_MODEL_TYPE=${EVAL_MODEL_TYPE:-PT}
if [ "${EVAL_MODEL_TYPE}" = "FT" ]; then
  DEFAULT_MODEL=${FT_DEFAULT_MODEL}
  CHECKPOINT=${CHECKPOINT:-${EVAL_FT_CHECKPOINT}}
else
  DEFAULT_MODEL=${PT_DEFAULT_MODEL}
  CHECKPOINT=${CHECKPOINT:-${EVAL_CHECKPOINT}}
fi
MODEL=${MODEL:-${DEFAULT_MODEL}}
POLICY_PATH=${POLICY_PATH:-${VLA_OUTPUT_ROOT}/${MODEL}/checkpoints/${CHECKPOINT}/pretrained_model}

# ── FSQ decoder and inference settings ───────────────────────────────────────
SKILL_DECODER_END_THRESHOLD=${SKILL_DECODER_END_THRESHOLD:-0.5}
SKILL_DECODER_STATE_INDICES=${SKILL_DECODER_STATE_INDICES_OVERRIDE:-${CFG_SKILL_DECODER_STATE_INDICES}}
SKILL_DECODER_PRIOR_NOISE_RATIO=${SKILL_DECODER_PRIOR_NOISE_RATIO_OVERRIDE:-${CFG_SKILL_DECODER_PRIOR_NOISE_RATIO}}
INFERENCE_SKILL_MAX_LENGTH=${INFERENCE_SKILL_MAX_LENGTH_OVERRIDE:-${CFG_INFERENCE_SKILL_MAX_LENGTH}}
DETACH_ACTION_PREFIX_GRAD=${DETACH_ACTION_PREFIX_GRAD_OVERRIDE:-${CFG_DETACH_ACTION_PREFIX_GRAD}}

# Eval uses simulated raw images; the FSQ decoder runs its frozen DINO internally.
if [ ! -d "${POLICY_PATH}" ]; then
  echo "Policy checkpoint not found: ${POLICY_PATH}" >&2
  exit 1
fi
if [ ! -f "${FSQ_CKPT}" ]; then
  echo "FSQ checkpoint not found: ${FSQ_CKPT}" >&2
  exit 1
fi

WANDB_PROJECT=${WANDB_PROJECT:-VLA_eval}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${MODEL}_${CHECKPOINT}_${TARGET_TASK}_fsq_eval}

# ── Output directory (inside skillVLA_eval folder) ───────────────────────────
if [ -d "${SUBMIT_DIR}/examples/libero/configs/skillVLA_eval" ]; then
  _EVAL_BASE="${SUBMIT_DIR}/examples/libero/configs/skillVLA_eval"
else
  _EVAL_BASE="${SUBMIT_DIR}"
fi
EVAL_OUT_DIR=${EVAL_OUT_DIR:-${_EVAL_BASE}/outputs/${WANDB_RUN_NAME}}
mkdir -p "${EVAL_OUT_DIR}"

echo "SkillVLA FSQ eval"
echo "  target     : ${TARGET_TASK}"
echo "  task_ids   : ${TASK_IDS}"
echo "  episodes   : ${N_EPISODES} per task"
echo "  videos     : ${MAX_VIDEOS_PER_TASK} per task, stride=${VIDEO_FRAME_STRIDE}, fps=${VIDEO_FPS}"
echo "  skill html : ${SKILL_HTML}, train_samples=${SKILL_HTML_TRAIN_SAMPLES}"
echo "  policy     : ${POLICY_PATH}"
echo "  fsq        : ${FSQ_CKPT}"
echo "  prior mix r: ${SKILL_DECODER_PRIOR_NOISE_RATIO}"
echo "  detach prefix grad: ${DETACH_ACTION_PREFIX_GRAD}"

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/python -m lerobot.scripts.lerobot_skillvla_eval \
  --policy.path=${POLICY_PATH} \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --env.fps=20 \
  --eval.batch_size=${EVAL_BATCH_SIZE} \
  --eval.n_episodes=${N_EPISODES} \
  --eval.max_videos_per_task=${MAX_VIDEOS_PER_TASK} \
  --eval.video_frame_stride=${VIDEO_FRAME_STRIDE} \
  --eval.video_fps=${VIDEO_FPS} \
  --eval.skill_html=${SKILL_HTML} \
  --eval.skill_html_train_samples=${SKILL_HTML_TRAIN_SAMPLES} \
  --eval.skill_html_skill_latents_path=${SKILL_LATENTS_PATH} \
  --eval.skill_html_raw_dataset_dir=${RAW_DATASET_DIR} \
  --eval.skill_html_image_key=${IMAGE_KEY} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.device=cuda \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=false \
  --policy.vae_decoder_path=${FSQ_CKPT} \
  --policy.skill_decoder_image_model_name=${IMAGE_MODEL_PATH} \
  --policy.skill_latent_dim=${FSQ_DIM} \
  --policy.skill_fsq_levels=${FSQ_LEVELS_ARG} \
  --policy.skill_predictor_num_embeddings=${FSQ_NUM_EMBEDDINGS} \
  --policy.skill_decoder_end_threshold=${SKILL_DECODER_END_THRESHOLD} \
  --policy.skill_decoder_state_indices=${SKILL_DECODER_STATE_INDICES} \
  --policy.use_fsq_latent_suffix=${USE_FSQ_LATENT_SUFFIX} \
  --policy.inference_skill_max_length=${INFERENCE_SKILL_MAX_LENGTH} \
  --policy.detach_action_prefix_grad=${DETACH_ACTION_PREFIX_GRAD} \
  --env.task_ids=${TASK_IDS} \
  --env.max_parallel_tasks=${MAX_PARALLEL_TASKS} \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --output_dir="${EVAL_OUT_DIR}" \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
  ${EXTRA_ARGS:-}

echo "DONE"
