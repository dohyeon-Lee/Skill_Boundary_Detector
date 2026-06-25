#!/usr/bin/env bash
# Shared DINO-preparation helper for the build_data pipeline (sourced, not executed).
#
# The config (train_skillVLA_config.py) declares the required (grid × camera) DINO set in
# DINO_REQUIRED ("grid:camera;grid:camera;..."), the slice-from / generate-into base in
# DINO_BASE_ROOT ({base|source}_DINO), and the per-grid work dirs under DINO_ROOT.
#
#   dino_prepare_slice  — for every required (grid × camera): VALIDATE it exists under
#                         DINO_BASE_ROOT (errors loudly if missing — no silent fallback) and
#                         slice it into the work dir. Login node (CPU). In generate mode it
#                         uses --identity (base keyed by the source's OWN episode index) and,
#                         if the base hasn't been computed yet, points the user at the master
#                         submit (which runs the GPU dino-prep job).
#
# safe_key mirrors prepare_dino_for_skillvla.safe_key: "observation.images.image" → "observation_images_image".

dino_safe_key () { echo "$1" | sed 's#[./]#_#g'; }

# Iterate DINO_REQUIRED as (grid, camera) pairs, calling `cb grid camera`.
dino_for_each_required () {
  local cb="$1" req grid cam
  local IFS=';'
  for req in ${DINO_REQUIRED}; do
    [ -z "${req}" ] && continue
    grid="${req%%:*}"; cam="${req#*:}"
    "${cb}" "${grid}" "${cam}"
  done
}

_dino_slice_one () {
  local grid="$1" cam="$2" extra=()
  [ "${DINO_GENERATE}" = "true" ] && extra=(--identity)   # base keyed by source's own ep index
  local safe; safe="$(dino_safe_key "${cam}")"
  local base_grid="${DINO_BASE_ROOT}/pg${grid}"
  if [ ! -d "${base_grid}/${safe}" ]; then
    if [ "${DINO_GENERATE}" = "true" ]; then
      echo "ERROR: source DINO not generated yet: ${base_grid}/${safe}/  (grid=${grid}, camera=${cam})" >&2
      echo "       run submit_build_all.sh — it submits the GPU dino-prep job for generate mode." >&2
    else
      echo "ERROR: required DINO missing from base '${DINO_BASE_DATASET}'." >&2
      echo "       need: ${base_grid}/${safe}/   (grid=${grid}, camera=${cam})" >&2
      echo "       → precompute it (configs/generate_training_dataset) or set 'dino_base_dataset: generate'." >&2
    fi
    exit 1
  fi
  "${BOOTSTRAP_PYTHON}" "${SRC_DIR}/prepare_dino_for_skillvla.py" \
    --dataset_dir   "${RAW_DATASET_DIR}" \
    --base_dino_dir "${base_grid}" \
    --dst_dir       "${DINO_ROOT}/pg${grid}" \
    --image_keys    "${cam}" \
    --mode          "${DINO_COPY_MODE}" \
    ${extra[@]+"${extra[@]}"}
}

dino_prepare_slice () {
  echo "Slice per-episode DINO  (base: ${DINO_BASE_ROOT}; required: ${DINO_REQUIRED})"
  dino_for_each_required _dino_slice_one
}
