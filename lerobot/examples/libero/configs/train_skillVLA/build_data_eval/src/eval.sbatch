#!/usr/bin/env bash
# SkillVLA-data evals off final artifacts → {run_dir}/eval/{dino,skillset,fsq_patch,fsq_recon}/
#SBATCH --job-name=skillVLA_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
mkdir -p logs

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # build_data_eval
SRC_DIR="${SUBMIT_DIR}/src"
BUILD_SRC_DIR="${SUBMIT_DIR}/../build_data/src"   # train_skillVLA_config.py lives with build_data
CONFIG_PATH="${TRAIN_SKILLVLA_CONFIG:-${SUBMIT_DIR}/../build_data/train_skillVLA_config.yaml}"
EVAL_CONFIG_PATH="${EVAL_CONFIG:-${SUBMIT_DIR}/eval_config.yaml}"
SOURCE_DATASET="${SOURCE_DATA:-}"

BOOTSTRAP_PYTHON="${SUBMIT_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

if [ -n "${SOURCE_DATASET}" ]; then
  eval "$("${BOOTSTRAP_PYTHON}" "${BUILD_SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --dataset "${SOURCE_DATASET}" --shell)"
else
  eval "$("${BOOTSTRAP_PYTHON}" "${BUILD_SRC_DIR}/train_skillVLA_config.py" --config "${CONFIG_PATH}" --shell)"
fi
eval "$("${BOOTSTRAP_PYTHON}" "${SRC_DIR}/eval_config.py" --config "${EVAL_CONFIG_PATH}" --shell)"

# eval_run_dir may point the eval at a different source dataset than the build_data yaml,
# but SKILLSET_DIR (boundary curves) still comes from the build_data source. The seg folder
# name is source-independent, so swap only the source component so the overlaid curve
# matches the episodes actually being evaluated.
EVAL_SOURCE="$(basename "${RAW_DATASET_DIR}")"
if [[ "${SKILLSET_DIR}" == *"/_work/"* && "${SKILLSET_DIR%%/_work/*}" != */"${EVAL_SOURCE}" ]]; then
  SKILLSET_DIR="$(dirname "${SKILLSET_DIR%%/_work/*}")/${EVAL_SOURCE}/_work/${SKILLSET_DIR#*/_work/}"
  echo "SKILLSET_DIR rewritten for eval source '${EVAL_SOURCE}': ${SKILLSET_DIR}"
fi

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/runtime_env.sh"
export LD_LIBRARY_PATH=""

echo "START SKILLVLA EVAL"
echo "Node     : ${SLURMD_NODENAME:-local}"
echo "skillvla : ${SKILLVLA_DATASET_DIR}"
echo "out      : ${EVAL_DIR}"
nvidia-smi || true

RUN_ARGS=()
[ "${EVAL_RUN_DINO}" = "true" ]      && RUN_ARGS+=(--run_dino)
[ "${EVAL_RUN_SKILLSET}" = "true" ]  && RUN_ARGS+=(--run_skillset)
[ "${EVAL_RUN_FSQ_PATCH}" = "true" ] && RUN_ARGS+=(--run_fsq_patch)
[ "${EVAL_RUN_FSQ_RECON}" = "true" ] && RUN_ARGS+=(--run_fsq_recon)

TASK_ARGS=()
[ -n "${EVAL_TASK_IDS}" ] && TASK_ARGS=(--task_ids ${EVAL_TASK_IDS})

PYTHONPATH="${PROJECT_ROOT}/lerobot/src" \
  python -u "${SRC_DIR}/run_eval.py" \
    --skillvla_dir   "${SKILLVLA_DATASET_DIR}" \
    --dino_model_path "${DINO_MODEL_PATH:-${PROJECT_ROOT}/models/dinov3-vits16}" \
    --fsq_model      "${FSQ_COPY_PATH}" \
    --dataset_dir    "${RAW_DATASET_DIR}" \
    --image_key      "${DINO_IMAGE_KEY:-observation.images.image}" \
    --out_dir        "${EVAL_DIR}" \
    --boundary_curves_dir "${SKILLSET_DIR}/curves" \
    "${RUN_ARGS[@]}" \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
    --n_episodes     "${EVAL_N_EPISODES}" \
    --n_samples      "${EVAL_N_SAMPLES}" \
    --max_entries    "${EVAL_MAX_ENTRIES}" \
    --n_action_steps "${EVAL_N_ACTION_STEPS}" \
    --end_threshold  "${EVAL_END_THRESHOLD}" \
    --thumb_size     "${EVAL_THUMB_SIZE}" \
    --seed           "${EVAL_SEED}"

echo "DONE → ${EVAL_DIR}"
