#!/usr/bin/env bash

# Shared visual-backbone presets for FSQ data generation scripts.
# VISUAL_BACKBONE choices:
#   dinov2_small
#   dinov3_vits16        (also accepts dinov3_vits14 as an alias)
#   dinov3_convnext_small

resolve_visual_backbone() {
  : "${HOMEDIR:?}" "${PROJDIR:?}"
  VISUAL_BACKBONE="${VISUAL_BACKBONE:-dinov2_small}"

  case "${VISUAL_BACKBONE}" in
    dinov2_small)
      IMAGE_MODEL_REPO="facebook/dinov2-small"
      IMAGE_MODEL_DIR="dinov2-small"
      IMAGE_FEATURE_TAG="dinov2_small"
      IMAGE_EXP_TAG="dino"
      ;;
    dinov3_vits16|dinov3_vits14)
      IMAGE_MODEL_REPO="facebook/dinov3-vits16-pretrain-lvd1689m"
      IMAGE_MODEL_DIR="dinov3-vits16"
      IMAGE_FEATURE_TAG="dinov3_vits16"
      IMAGE_EXP_TAG="dinov3_vits16"
      ;;
    dinov3_convnext_small)
      IMAGE_MODEL_REPO="facebook/dinov3-convnext-small-pretrain-lvd1689m"
      IMAGE_MODEL_DIR="dinov3-convnext-small"
      IMAGE_FEATURE_TAG="dinov3_convnext_small"
      IMAGE_EXP_TAG="dinov3_convnext_small"
      ;;
    *)
      echo "Unknown VISUAL_BACKBONE=${VISUAL_BACKBONE}"
      echo "Use one of: dinov2_small, dinov3_vits16, dinov3_convnext_small"
      exit 1
      ;;
  esac

  IMAGE_MODEL_PATH="${HOMEDIR}${PROJDIR}/models/${IMAGE_MODEL_DIR}"

  export VISUAL_BACKBONE IMAGE_MODEL_REPO IMAGE_MODEL_DIR IMAGE_FEATURE_TAG IMAGE_EXP_TAG
  export IMAGE_MODEL_PATH
}
