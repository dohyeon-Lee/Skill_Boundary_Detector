#!/usr/bin/env bash
# One argv source for initial generation, curve collection/cached segmentation
# and missing-episode retries. Missing exports fail instead of using defaults.
skillvla_skillset_args() {
  : "${SKILLSET_CONTRACT_JSON:?Old/incomplete config snapshot; resolve the current build configuration first}"
  SKILLSET_CONTRACT_ARGS=(
    --probe_generation "${SKILLSET_PROBE_GENERATION:?}"
    --gaussian_sampling "${SKILLSET_GAUSSIAN_SAMPLING:?}"
    --gaussian_seed "${SKILLSET_GAUSSIAN_SEED:?}"
    --denoise_steps "${SKILLSET_DENOISE_STEPS:?}"
    --denoise_output "${SKILLSET_DENOISE_OUTPUT:?}"
    --score_metric "${SKILLSET_SCORE_METRIC:?}"
    --gmm_covariance "${SKILLSET_GMM_COVARIANCE:?}"
    --gmm_n_init "${SKILLSET_GMM_N_INIT:?}"
    --gmm_max_iter "${SKILLSET_GMM_MAX_ITER:?}"
    --min_effective_samples "${SKILLSET_MIN_EFFECTIVE_SAMPLES:?}"
    --score_power "${SKILLSET_SCORE_POWER:?}"
    --terminal_mask_frames "${SKILLSET_TERMINAL_MASK_FRAMES:?}"
    --noise_scheduler_type "${SKILLSET_NOISE_SCHEDULER_TYPE:?}"
    --num_inference_steps "${SKILLSET_NUM_INFERENCE_STEPS:?}"
    --seed "${SKILLSET_SEED:?}"
    --segmentation_contract_json "${SKILLSET_CONTRACT_JSON}"
  )
  local pair name enabled
  for pair in "GAUSSIAN_INCLUDE_MEAN gaussian-include-mean" "GMM_WEIGHTED gmm-weighted" "PEAK_NMS peak-nms"; do
    read -r name enabled <<< "${pair}"
    name="SKILLSET_${name}"
    case "${!name:?}" in
      true) SKILLSET_CONTRACT_ARGS+=("--${enabled}") ;;
      false) SKILLSET_CONTRACT_ARGS+=("--no-${enabled}") ;;
      *) echo "Invalid boolean ${name}=${!name}" >&2; return 1 ;;
    esac
  done
}

skillvla_check_contract() {
  : "${SKILLSET_CONTRACT_JSON:?Old/incomplete config snapshot; resolve the current build configuration first}"
  local snap=""
  [ "${FSQ_SNAP_TO_SUPPORTED:-false}" != true ] || snap="${FSQ_SNAP_REFERENCE:-}"
  "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/lerobot/examples/libero/configs/train_skills/src/sbd_build_contract.py" \
    --contract-json "${SKILLSET_CONTRACT_JSON}" --skillset-dir "${SKILLSET_DIR}" \
    --run-dir "${SKILLVLA_RUN_DIR}" --stage "$1" \
    --threshold-reference "${SKILLSET_GLOBAL_THRESHOLD_SOURCE:-}" --snap-reference "${snap}"
}
