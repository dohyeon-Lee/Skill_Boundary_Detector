#!/usr/bin/env bash
#SBATCH --job-name=pi05_eval

# SBATCH --partition=asus_pro6000 #,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090
# SBATCH --exclude=node19,node13,node18,node16,node08,node10,node21,node14,node04,node05,node31
# SBATCH --qos=pro6000_qos

#SBATCH --partition=debug
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
N_EPISODES=${N_EPISODES:-1}
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
EVAL_STAGE=${EVAL_STAGE:-PT}  # PT or FT
CHECKPOINT=${CHECKPOINT:-015000}

PT_DATASET=${PT_DATASET:-libero_10_op1_10}
PT_BATCH_SIZE=${PT_BATCH_SIZE:-32}
PT_EXP=${PT_EXP:-exp1}

FT_DATASET=${FT_DATASET:-libero_10_op2_1}
FT_BATCH_SIZE=${FT_BATCH_SIZE:-32}
FT_EXP=${FT_EXP:-}
PT_CHECKPOINT=${PT_CHECKPOINT:-070000}

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

WANDB_PROJECT=VLA_eval
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${MODEL}_${CHECKPOINT}_${TARGET_TASK}}

echo "pi05 eval"
echo "  target     : ${TARGET_TASK}"
echo "  task_ids   : ${TASK_IDS}"
echo "  episodes   : ${N_EPISODES}"
echo "  policy     : ${POLICY_PATH}"

PYTHONPATH=${HOMEDIR}${PROJDIR}/lerobot/src \
  ${HOMEDIR}${PROJDIR}/.venv/bin/lerobot-eval \
  --policy.path=${POLICY_PATH} \
  --env.type=libero \
  --env.task=${TARGET_TASK} \
  --eval.batch_size=${EVAL_BATCH_SIZE} \
  --eval.n_episodes=${N_EPISODES} \
  --policy.device=cuda \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --env.task_ids=${TASK_IDS} \
  --wandb_project ${WANDB_PROJECT} \
  --job_name ${WANDB_RUN_NAME} \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'
