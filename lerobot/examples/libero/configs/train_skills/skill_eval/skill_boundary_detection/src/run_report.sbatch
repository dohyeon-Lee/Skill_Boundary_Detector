#!/usr/bin/env bash
#SBATCH --job-name=LIBERO_SBD_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

EVAL_DIR="${LIBERO_SBD_REPORT_DIR:?launch with submit_report.sh}"
CONFIG="${LIBERO_SBD_REPORT_CONFIG:?missing report config}"
BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/src/config.py" --config "${CONFIG}" --shell)"

cd "${PROJECT_ROOT}/lerobot"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/libero/configs/src/runtime_env.sh"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-libero_sbd}_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

TASK_IDS=( ${DP_BUILT_REPORT_TASK_IDS//,/ } )
TARGET_TASK_ARGS=()
if [ "${DP_BUILT_REPORT_TASK_ID_SPACE}" = suite ]; then
  TARGET_TASK_ARGS+=(--target_task "${DP_BUILT_REPORT_TARGET_TASK}")
fi
GMM_ARGS=()
if [ -n "${LIBERO_SBD_GMM_RAW_DIR:-}" ]; then
  GMM_ARGS+=(--gmm_raw_dir "${LIBERO_SBD_GMM_RAW_DIR}" --gmm_components 5)
fi

PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/src" \
  python -u examples/libero/configs/train_skills/skill_eval/src/dp_skillset_eval.py \
    --skillset_dir "${DP_ABLATION_SKILLSET_DIR}" \
    --image_key "${DP_BUILT_REPORT_IMAGE_KEY}" \
    --out_dir "${DP_BUILT_REPORT_OUTPUT_DIR}" \
    --out_html index.html \
    --title "LIBERO ${DP_BUILT_REPORT_TARGET_TASK} · DP ck${DP_BUILT_REPORT_CHECKPOINT} · built SBD boundaries" \
    --task_id_space "${DP_BUILT_REPORT_TASK_ID_SPACE}" \
    "${TARGET_TASK_ARGS[@]}" \
    --task_ids "${TASK_IDS[@]}" \
    --n_episodes "${DP_BUILT_REPORT_N_EPISODES}" \
    --episode_video \
    --episode_video_clips \
    --hide_start_end_frames \
    --hide_bic_graph \
    --hide_gripper_graph \
    "${GMM_ARGS[@]}"

echo "LIBERO SBD visualization complete: ${DP_BUILT_REPORT_OUTPUT_DIR}/index.html"
