#!/usr/bin/env bash
#SBATCH --job-name=ABC_SBD_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

EVAL_DIR="${ABC_SBD_REPORT_DIR:?launch with submit.sh}"
BUILD_CONFIG="${ABC_BUILD_DATA_CONFIG:?missing build config}"
REPORT_EPISODES="${ABC_SBD_REPORT_EPISODES:-5}"
BOUNDARY_SOURCE="${ABC_SBD_BOUNDARY_SOURCE:-skillset}"
REPORT_OUTPUT="${ABC_SBD_REPORT_OUTPUT:?missing report output directory}"
BOOTSTRAP_PYTHON="${EVAL_DIR}/../../../../../../../.venv/bin/python"
[ -x "${BOOTSTRAP_PYTHON}" ] || BOOTSTRAP_PYTHON=python3
eval "$("${BOOTSTRAP_PYTHON}" "${EVAL_DIR}/../../src/train_skills_config.py" \
  --config "${BUILD_CONFIG}" --stage build_data --shell)"

cd "${LEROBOT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/lerobot/examples/abc/configs/src/runtime_env.sh"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-abc_sbd}_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

EXTRA_ARGS=()
GMM_ARGS=()
if [ -n "${ABC_SBD_GMM_RAW_DIR:-}" ]; then
  GMM_ARGS+=(--gmm_raw_dir "${ABC_SBD_GMM_RAW_DIR}" --gmm_components 5)
fi
case "${BOUNDARY_SOURCE}" in
  skillset)
    REPORT_TITLE="ABC grab_clutter · DP ck${DP_CHECKPOINT} · built SBD boundaries"
    ;;
  denoising_gain)
    REPORT_TITLE="ABC grab_clutter · DP ck${DP_CHECKPOINT} · denoising-gain global-mean boundaries"
    EXTRA_ARGS+=(--boundary_source denoising_gain --hide_cos_graph)
    ;;
  all_peaks)
    REPORT_TITLE="ABC grab_clutter · DP ck${DP_CHECKPOINT} · all SBD local peaks (no height threshold)"
    EXTRA_ARGS+=(--boundary_source all_peaks)
    ;;
  power_half_mean)
    REPORT_TITLE="ABC grab_clutter · DP ck${DP_CHECKPOINT} · SBD power=0.5 global-mean boundaries"
    EXTRA_ARGS+=(--boundary_source power_half_mean)
    ;;
  power_one_initial_motion_mask)
    REPORT_TITLE="ABC grab_clutter · DP ck${DP_CHECKPOINT} · SBD power=1 · reset-aware motion-start+100 mask"
    EXTRA_ARGS+=(--boundary_source power_one_initial_motion_mask)
    ;;
  *)
    echo "Unsupported ABC_SBD_BOUNDARY_SOURCE: ${BOUNDARY_SOURCE}" >&2
    exit 2
    ;;
esac
TASK_IDS="${ABC_TASK_INDICES//,/ }"
PYTHONPATH="${PROJECT_ROOT}/lerobot/src:${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/skill_eval/src" \
  python -u examples/libero/configs/train_skills/skill_eval/src/dp_skillset_eval.py \
    --skillset_dir "${SKILLSET_DIR}" \
    --image_key observation.images.top \
    --out_dir "${REPORT_OUTPUT}" \
    --out_html index.html \
    --title "${REPORT_TITLE}" \
    --task_id_space dataset \
    --task_ids ${TASK_IDS} \
    --n_episodes "${REPORT_EPISODES}" \
    --episode_video \
    --hide_start_end_frames \
    --hide_bic_graph \
    --hide_gripper_graph \
    "${GMM_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "ABC SBD visualization complete: ${REPORT_OUTPUT}/index.html"
