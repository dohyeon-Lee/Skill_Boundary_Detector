#!/usr/bin/env bash
# Submit SkillVLA Stage-1 VSA training (policy.type=skill_expert).
# Usage: ./submit_train.sh [arch0|arch0_skill|arch0_skill_chunk|arch1|arch1_skill|arch1_skill_chunk|arch2|arch2_skill|arch2_skill_chunk|arch3|arch3_skill|arch3_skill_chunk|arch4|arch4_skill|arch4_skill_chunk|arch5|arch5_skill|arch5_skill_chunk|arch6|arch6_skill|arch6_skill_chunk]
# Also accepts fullskill_1/2, fullskill_vis_{1,2}_{firstlayer,lastlayer},
# fullskill_bspline_{2,3,4,5,6}_{firstlayer,lastlayer} (split selected by visual_bridge_last_n_layers),
# fullskill_bspline_3_{firstlayer,lastlayer}_{lastvis,lastvis_first,lastvis_last},
# residual_patch_1..6 / direct_patch_1..6, optionally suffixed _dGD.
# Omit the argument to use architecture.name.
#   (login) resolve config + check the skillvla dataset → sbatch train.sbatch
# The skillvla dataset comes from configs/train_skillVLA/build_data (run that first if missing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # PT/VSA
STAGE1_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # shared Stage-1 code: PT/src
SRC_DIR="${STAGE1_DIR}/src"
CONFIG_PATH="${STAGE1_TRAIN_CONFIG:-${SCRIPT_DIR}/vsa_train_config.yaml}"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [arch0|arch0_skill|arch0_skill_chunk|arch1|arch1_skill|arch1_skill_chunk|arch2|arch2_skill|arch2_skill_chunk|arch3|arch3_skill|arch3_skill_chunk|arch4|arch4_skill|arch4_skill_chunk|arch5|arch5_skill|arch5_skill_chunk|arch6|arch6_skill|arch6_skill_chunk]" >&2
  exit 2
fi
ARCHITECTURE_OVERRIDE="${1:-${STAGE1_ARCHITECTURE_OVERRIDE:-}}"
ARCHITECTURE_OVERRIDE="${ARCHITECTURE_OVERRIDE,,}"
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  case "${ARCHITECTURE_OVERRIDE}" in
    vanila|vanila_nouv|vanila_last|vanila_nouv_last|vanila_vision_only) ;;
    fullskill_1|fullskill_2|fullskill_vis_1|fullskill_vis_[12]_firstlayer|fullskill_vis_[12]_lastlayer|fullskill_bspline_[23456]_firstlayer|fullskill_bspline_[23456]_lastlayer) ;;
    fullskill_bspline_3_lastlayer_lastvis|fullskill_bspline_3_lastlayer_lastvis_first|fullskill_bspline_3_lastlayer_lastvis_last) ;;
    fullskill_bspline_3_lastlayer_lastvis_current|fullskill_bspline_3_lastlayer_lastvis_current_preskill|fullskill_bspline_3_lastlayer_lastvis_current_none) ;;
    uv_fullskill_bspline_3_lastlayer_lastvis|uv_fullskill_bspline_3_lastlayer_lastvis_current|uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill|uv_fullskill_bspline_3_lastlayer_lastvis_current_none) ;;
    uv_fullskill_bspline_3_lastlayer_lastvis_diffaux|uv_fullskill_bspline_3_lastlayer_lastvis_current_diffaux|uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux|uv_fullskill_bspline_3_lastlayer_lastvis_current_none_diffaux) ;;
    uv_fullskill_bspline_3_lastlayer_lastvis_last|uv_fullskill_bspline_3_lastlayer_lastvis_diffaux_last|uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux_last|uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_last) ;;
    uv_bspline_lastlayer_current_preskill_diffaux) ;;
    uv_bspline_skill_learnable|uv_bspline_skill_denoising|uv_bspline_skill_learnable_query|uv_bspline_skill_denoising_none) ;;
    uv_bspline_skill_learnable_wovis|uv_bspline_skill_denoising_none_wovis) ;;
    uv_bspline_skill_learnable_nouv|uv_bspline_skill_learnable_wovis_nouv|uv_bspline_skill_denoising_none_nouv|uv_bspline_skill_denoising_none_wovis_nouv) ;;
    uv_bspline_lastlayer_diffaux3|uv_bspline_lastlayer_diffaux3_last|uv_bspline_lastlayer_current_preskill_diffaux3|uv_bspline_lastlayer_current_preskill_diffaux3_last) ;;
    uv_bspline_lastlayer_diffaux4|uv_bspline_lastlayer_diffaux4_uvtoken|uv_bspline_lastlayer_current_preskill_diffaux4|uv_bspline_lastlayer_current_preskill_diffaux4_uvtoken) ;;
    uv_bspline_lastlayer_diffaux4_last|uv_bspline_lastlayer_current_preskill_diffaux4_last) ;;
    uv_bspline_lastlayer_diffaux4_start|uv_bspline_lastlayer_current_preskill_diffaux4_start) ;;
    uv_bspline_lastlayer_diffaux4_start_last|uv_bspline_lastlayer_current_preskill_diffaux4_start_last) ;;
    uv_bspline_lastlayer_diffaux5_last|uv_bspline_lastlayer_current_preskill_diffaux5_last) ;;
    uv_bspline_lastlayer_diffaux5|uv_bspline_lastlayer_current_preskill_diffaux5) ;;
    uv_bspline_lastlayer_diffaux6_last|uv_bspline_lastlayer_current_preskill_diffaux6_last) ;;
    uv_bspline_lastlayer_diffaux6|uv_bspline_lastlayer_current_preskill_diffaux6) ;;
    uv_bspline_lastlayer_current_preskill_diffaux2_detach|uv_bspline_lastlayer_current_preskill_diffaux2_attach|uv_bspline_lastlayer_current_preskill_diffaux2_detach_topw1|uv_bspline_lastlayer_current_preskill_diffaux2_attach_topw1) ;;
    fullskill_bspline_3_firstlayer_lastvis|fullskill_bspline_3_firstlayer_lastvis_first|fullskill_bspline_3_firstlayer_lastvis_last) ;;
    residual_patch_[1-6]|direct_patch_[1-6]|residual_patch_[1-6]_dgd|direct_patch_[1-6]_dgd) ;;
    arch0|arch0_skill|arch0_skill_chunk|arch1|arch1_skill|arch1_skill_chunk|arch2|arch2_skill|arch2_skill_chunk|arch3|arch3_skill|arch3_skill_chunk|arch4|arch4_skill|arch4_skill_chunk|arch5|arch5_skill|arch5_skill_chunk|arch6|arch6_skill|arch6_skill_chunk) ;;
    *)
      echo "Unknown Stage-1 architecture: ${ARCHITECTURE_OVERRIDE}" >&2
      exit 2
      ;;
  esac
fi

# Freeze the config so this job ignores later edits to the repo yaml (see configs/src/snapshot_config.sh).
_lib="$(dirname "${CONFIG_PATH}")"; while [ ! -f "${_lib}/src/snapshot_config.sh" ]; do _lib="$(dirname "${_lib}")"; done
source "${_lib}/src/snapshot_config.sh"
source "${_lib}/src/submit_job.sh"   # sbatch, or a local run where the server has no Slurm
CONFIG_PATH="$(snapshot_config "${CONFIG_PATH}")"

BOOTSTRAP_PYTHON="${STAGE1_DIR}/../../../../../../.venv/bin/python"
if [ ! -x "${BOOTSTRAP_PYTHON}" ]; then
  BOOTSTRAP_PYTHON=python3
fi

CONFIG_ARGS=(--config "${CONFIG_PATH}" --shell)
if [ -n "${ARCHITECTURE_OVERRIDE}" ]; then
  CONFIG_ARGS+=(--architecture "${ARCHITECTURE_OVERRIDE}")
fi
if ! BOOTSTRAP_EXPORTS="$(
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/stage1_train_config.py" "${CONFIG_ARGS[@]}"
)"; then
  echo "Stage-1 configuration bootstrap failed; no job was submitted." >&2
  exit 1
fi
eval "${BOOTSTRAP_EXPORTS}"
: "${SKILLVLA_DATASET_DIR:?Stage-1 bootstrap did not export SKILLVLA_DATASET_DIR}"

if [ ! -e "${SKILLVLA_DATASET_DIR}" ]; then
  echo "Missing skillvla dataset: ${SKILLVLA_DATASET_DIR}" >&2
  echo "Build it first: configs/train_skillVLA/build_data/submit_build_all.sh" >&2
  exit 1
fi

SBATCH_ARGS=(
  --partition="${TRAIN_PARTITION}"
  --qos="${TRAIN_QOS}"
  --gres="${TRAIN_GRES}"
  --cpus-per-task="${TRAIN_CPUS_PER_TASK}"
  --mem="${TRAIN_MEM}"
  --time="${TRAIN_TIME}"
)
if [ -n "${TRAIN_NODELIST}" ]; then
  SBATCH_ARGS+=(--nodelist="${TRAIN_NODELIST}")
fi
if [ -n "${TRAIN_EXCLUDE_NODES}" ]; then
  SBATCH_ARGS+=(--exclude="${TRAIN_EXCLUDE_NODES}")
fi

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Submit Stage-1 ${ARCHITECTURE_LABEL} (skill_expert)"
echo "  run      : ${PT_RUN_NAME}"
echo "  dataset  : ${SKILLVLA_DATASET_DIR}"
echo "  expert   : full fine-tuning from ${PI_BASE}"
echo "  output   : ${PT_OUTPUT_DIR}"
echo "  slurm    : partition=${TRAIN_PARTITION} qos=${TRAIN_QOS} gres=${TRAIN_GRES} mem=${TRAIN_MEM}"

STAGE1_TRAIN_CONFIG="${CONFIG_PATH}" \
STAGE1_ARCHITECTURE_OVERRIDE="${ARCHITECTURE_OVERRIDE}" \
STAGE1_SRC_DIR="${SRC_DIR}" \
  submit_job "${SBATCH_ARGS[@]}" "${SRC_DIR}/train.sbatch"
